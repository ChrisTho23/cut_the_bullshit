# FROM: https://huggingface.co/blog/welcome-openai-gpt-oss#local-inference

import logging
import time
from pathlib import Path
from typing import Literal

import nnsight
import pandas as pd
import torch as t
from fire import Fire
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoTokenizer

from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def single_message_inference(
    lm: LanguageModel,
    tok: AutoTokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    prompt_len = inputs["input_ids"].shape[-1]

    with lm.generate(inputs, max_new_tokens=max_new_tokens):
        hidden_state_per_layer =  [nnsight.list().save() for _ in range(lm.config.num_hidden_layers)]
        with lm.model.layers.all():
            for i in range(lm.config.num_hidden_layers):
                hidden_state_per_layer[i].append(lm.model.layers[i].output[0])
        generated_ids = lm.generator.output.save()

    completion_token_ids = generated_ids[0][prompt_len:].cpu()
    thinking_token_idxs, final_token_idxs = utils.parser(completion_token_ids)
    thinking = tok.decode(completion_token_ids[thinking_token_idxs])
    final = tok.decode(completion_token_ids[final_token_idxs]) if final_token_idxs is not None else None
    generations = {
        "thinking": thinking,
        "final": final,
        "total": f"Thinking: {thinking} Final answer: {final}"
    }
    
    res_acts = t.stack([t.cat(hidden_state_per_layer[i].value, dim=0).detach().cpu() for i in range(lm.config.num_hidden_layers)], dim=0)
    prompt_res_acts = res_acts[:, :prompt_len] # num_hidden_layers, num_tokens, d_model
    gen_res_acts = res_acts[:, prompt_len:] # num_hidden_layers, num_tokens, d_model
    total_gen_tokens = gen_res_acts.shape[1]
    
    # Expand thinking indices to include start tokens (3 tokens: <|channel|>analysis<|message|>) and stop token (1 token: <|end|>)
    start_thinking_tokens = 3  # <|channel|>analysis<|message|>
    stop_thinking_tokens = 1   # <|end|>
    
    thinking_indices_list = []
    thinking_indices_list.append(t.arange(max(0, thinking_token_idxs[0] - start_thinking_tokens), thinking_token_idxs[0]))
    thinking_indices_list.append(thinking_token_idxs)
    stop_end = min(thinking_token_idxs[-1] + 1 + stop_thinking_tokens, total_gen_tokens)
    if thinking_token_idxs[-1] + 1 < total_gen_tokens:
        thinking_indices_list.append(t.arange(thinking_token_idxs[-1] + 1, stop_end))
    
    thinking_with_special_idxs = t.cat(thinking_indices_list)

    # Expand final indices to include start tokens (5 tokens: <|start|>assistant<|channel|>final<|message|>) and stop token (1 token: <|return|>)
    start_final_tokens = 5  # <|start|>assistant<|channel|>final<|message|>
    stop_final_tokens = 1   # <|return|>
    if final_token_idxs is not None:
        final_indices_list = []
        final_indices_list.append(t.arange(max(0, final_token_idxs[0] - start_final_tokens), final_token_idxs[0]))
        final_indices_list.append(final_token_idxs)
        stop_end = min(final_token_idxs[-1] + 1 + stop_final_tokens, total_gen_tokens)
        if final_token_idxs[-1] + 1 < total_gen_tokens:
            final_indices_list.append(t.arange(final_token_idxs[-1] + 1, stop_end))
        
        final_with_special_idxs = t.cat(final_indices_list)
    else:
        final_with_special_idxs = None

    avg_res_acts_per_layer_thinking = gen_res_acts[:, thinking_with_special_idxs, :].mean(dim=1)
    avg_res_acts_per_layer_final = gen_res_acts[:, final_with_special_idxs, :].mean(dim=1) if final_with_special_idxs is not None else None
    n_completion_tokens = len(thinking_with_special_idxs) + (len(final_with_special_idxs) if final_with_special_idxs is not None else 0)
    avg_res_acts_per_layer = avg_res_acts_per_layer_thinking * len(thinking_with_special_idxs) / n_completion_tokens
    if final_with_special_idxs is not None:
        avg_res_acts_per_layer += (avg_res_acts_per_layer_final * len(final_with_special_idxs)) / n_completion_tokens

    avg_acts = {
        "prompt": prompt_res_acts.mean(dim=1),
        "thinking": avg_res_acts_per_layer_thinking,
        "final": avg_res_acts_per_layer_final,
        "total": avg_res_acts_per_layer,
    }

    return generations, avg_acts

def main(
    model_id: Literal["openai/gpt-oss-20b", "openai/gpt-oss-120b"] = "openai/gpt-oss-20b",
    system_prompt_path: str = "src/data/system_prompt.txt",
    no_bs_system_prompt_path: str = "src/data/system_prompt_no_bs.txt",
    user_prompt_path: str = "src/data/user_prompts_coding_test.jsonl",
    max_new_tokens: int = 4096,
    output_dir: str = "src/data/outputs/test"
):
    logger.info("Reading user prompts...")
    user_prompts_df = pd.read_json(user_prompt_path, lines=True).copy()

    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LanguageModel(
        model_id,
        device_map="auto",
        torch_dtype=t.bfloat16,
    )

    logger.info("Loading system prompt...")
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    with open(no_bs_system_prompt_path, "r", encoding="utf-8") as f:
        no_bs_system_prompt = f.read().strip()

    user_prompts_df["prompt_plus_no_bs"] = user_prompts_df.apply(lambda row: str(row["prompt_plus"]) + " " + str(row["search_results"]), axis=1)

    thinking_plus, final_plus, total_plus = [], [], []
    prompt_activations_plus, thinking_activations_plus, final_activations_plus, activations_plus = [], [], [], []
    thinking_plus_no_bs, final_plus_no_bs, total_plus_no_bs = [], [], []
    prompt_activations_plus_no_bs, thinking_activations_plus_no_bs, final_activations_plus_no_bs, activations_plus_no_bs = [], [], [], []
    thinking_minus, final_minus, total_minus = [], [], []
    prompt_activations_minus, thinking_activations_minus, final_activations_minus, activations_minus = [], [], [], []

    logger.info("Running infernece on user prompts...")
    pbar = tqdm(total=len(user_prompts_df), desc="Running inference")
    for _, row in user_prompts_df.iterrows():
        plus = str(row["prompt_plus"])
        plus_no_bs = str(row["prompt_plus_no_bs"])
        minus = str(row["prompt_minus"])

        # plus case
        generations_plus, avg_acts_plus = single_message_inference(model, tokenizer, system_prompt, plus, max_new_tokens)
        
        thinking_plus.append(generations_plus["thinking"])
        final_plus.append(generations_plus["final"])
        total_plus.append(generations_plus["total"])
        prompt_activations_plus.append(avg_acts_plus["prompt"])
        thinking_activations_plus.append(avg_acts_plus["thinking"])
        final_activations_plus.append(avg_acts_plus["final"])
        activations_plus.append(avg_acts_plus["total"])
        # plus case + anti bs prompt
        generations_plus_no_bs, avg_acts_plus_no_bs = single_message_inference(model, tokenizer, no_bs_system_prompt, plus_no_bs, max_new_tokens)
        
        thinking_plus_no_bs.append(generations_plus_no_bs["thinking"])
        final_plus_no_bs.append(generations_plus_no_bs["final"])
        total_plus_no_bs.append(generations_plus_no_bs["total"])
        prompt_activations_plus_no_bs.append(avg_acts_plus_no_bs["prompt"])
        thinking_activations_plus_no_bs.append(avg_acts_plus_no_bs["thinking"])
        final_activations_plus_no_bs.append(avg_acts_plus_no_bs["final"])
        activations_plus_no_bs.append(avg_acts_plus_no_bs["total"])
        # minus case
        generations_minus, avg_acts_minus = single_message_inference(model, tokenizer, system_prompt, minus, max_new_tokens)
        
        thinking_minus.append(generations_minus["thinking"])
        final_minus.append(generations_minus["final"])
        total_minus.append(generations_minus["total"])
        prompt_activations_minus.append(avg_acts_minus["prompt"])
        thinking_activations_minus.append(avg_acts_minus["thinking"])
        final_activations_minus.append(avg_acts_minus["final"])
        activations_minus.append(avg_acts_minus["total"])

        pbar.update(1)
    pbar.close()

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()

    # Save generations
    output_csv_path = Path(output_dir, f"generations_{timestamp}.csv")

    user_prompts_df["thinking_plus"], user_prompts_df["final_plus"], user_prompts_df["total_plus"] = thinking_plus, final_plus, total_plus
    user_prompts_df["thinking_plus_no_bs"], user_prompts_df["final_plus_no_bs"], user_prompts_df["total_plus_no_bs"] = thinking_plus_no_bs, final_plus_no_bs, total_plus_no_bs
    user_prompts_df["thinking_minus"], user_prompts_df["final_minus"], user_prompts_df["total_minus"] = thinking_minus, final_minus, total_minus
    user_prompts_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved generations to {output_csv_path}...")

    # Save activations
    output_plus_prompt_pt_path = Path(output_dir, f"avg_resid_activations_plus_prompt_{timestamp}.pt")
    output_plus_thinking_pt_path = Path(output_dir, f"avg_resid_activations_plus_thinking_{timestamp}.pt")
    output_plus_final_pt_path = Path(output_dir, f"avg_resid_activations_plus_final_{timestamp}.pt")
    output_plus_pt_path = Path(output_dir, f"avg_resid_activations_plus_{timestamp}.pt")

    output_plus_no_bs_prompt_pt_path = Path(output_dir, f"avg_resid_activations_plus_no_bs_prompt_{timestamp}.pt")
    output_plus_no_bs_thinking_pt_path = Path(output_dir, f"avg_resid_activations_plus_no_bs_thinking_{timestamp}.pt")
    output_plus_no_bs_final_pt_path = Path(output_dir, f"avg_resid_activations_plus_no_bs_final_{timestamp}.pt")
    output_plus_no_bs_pt_path = Path(output_dir, f"avg_resid_activations_plus_no_bs_{timestamp}.pt")

    output_minus_prompt_pt_path = Path(output_dir, f"avg_resid_activations_minus_prompt_{timestamp}.pt")
    output_minus_thinking_pt_path = Path(output_dir, f"avg_resid_activations_minus_thinking_{timestamp}.pt")
    output_minus_final_pt_path = Path(output_dir, f"avg_resid_activations_minus_final_{timestamp}.pt")
    output_minus_pt_path = Path(output_dir, f"avg_resid_activations_minus_{timestamp}.pt")

    def save_activations_list(acts_list, path: Path):
        try:
            if all(isinstance(a, t.Tensor) for a in acts_list):
                t.save(t.stack(acts_list, dim=0), path)
            else:
                t.save(acts_list, path)
        except Exception:
            t.save(acts_list, path)

    save_activations_list(prompt_activations_plus, output_plus_prompt_pt_path)
    save_activations_list(thinking_activations_plus, output_plus_thinking_pt_path)
    save_activations_list(final_activations_plus, output_plus_final_pt_path)
    save_activations_list(activations_plus, output_plus_pt_path)

    save_activations_list(prompt_activations_plus_no_bs, output_plus_no_bs_prompt_pt_path)
    save_activations_list(thinking_activations_plus_no_bs, output_plus_no_bs_thinking_pt_path)
    save_activations_list(final_activations_plus_no_bs, output_plus_no_bs_final_pt_path)
    save_activations_list(activations_plus_no_bs, output_plus_no_bs_pt_path)

    save_activations_list(prompt_activations_minus, output_minus_prompt_pt_path)
    save_activations_list(thinking_activations_minus, output_minus_thinking_pt_path)
    save_activations_list(final_activations_minus, output_minus_final_pt_path)
    save_activations_list(activations_minus, output_minus_pt_path)

    logger.info("Saved activations...")

if __name__ == '__main__':
    Fire(main)
