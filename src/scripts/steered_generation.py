import logging
import pickle
import time
from pathlib import Path
from typing import List, Literal

import pandas as pd
import torch as t
from fire import Fire
from jaxtyping import Float
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoTokenizer

from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def steered_single_message_inference(
    lm: LanguageModel,
    tok: AutoTokenizer,
    system_prompt: str,
    user_prompt: str,
    steering_vec: Float[t.Tensor, "d_model"],
    layer_idx: int,
    max_new_tokens: int,
    alpha: float = 0.1,
    pad_token_id: int = 199999
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

    steering_vec = steering_vec.unsqueeze(0) # 1 d_model

    with lm.generate(inputs, max_new_tokens=max_new_tokens):
        with lm.model.layers.all():
            current_acts = lm.model.layers[layer_idx].output[0].clone()
            steered_acts = current_acts - alpha * steering_vec
            lm.model.layers[layer_idx].output[0] = steered_acts
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

    return generations

def main(
    model_id: Literal["openai/gpt-oss-20b", "openai/gpt-oss-120b"] = "openai/gpt-oss-20b",
    steering_tensor_path: str | Path = "src/data/outputs/train/bs_tensor_ppnbs_1755985687.141758.pkl",
    steering_tensor_key: str = "total",
    system_prompt_path: str = "src/data/system_prompt.txt",
    no_bs_system_prompt_path: str = "src/data/system_prompt_no_bs.txt",
    user_prompt_path: str = "src/data/user_prompts_coding_test.jsonl",
    max_new_tokens: int = 4096,
    output_dir: str = "src/data/outputs/test",
    layer_idx: int | None = None,
    prompt_type: Literal["plus", "minus", "no_bs"] = "plus",
    alpha: float = 1.0
):
    logger.info("Reading user prompts...")
    user_prompts_df = pd.read_json(user_prompt_path, lines=True).copy()

    logger.info("Loading steering tensor...")
    with open(steering_tensor_path, "rb") as f:
        data = pickle.load(f)
        bs_act_vector = data.get(steering_tensor_key) # (num_layer, d_model)

    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
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

    # Select system prompt and prompt column based on prompt_type
    if prompt_type == "no_bs":
        selected_system_prompt = no_bs_system_prompt
        prompt_column = "prompt_plus_no_bs"
    else:
        selected_system_prompt = system_prompt
        prompt_column = f"prompt_{prompt_type}"

    # Determine which layers to use for steering
    if layer_idx is not None:
        layers_to_steer = [layer_idx]
        logger.info(f"Using layer {layer_idx} for steering with prompt type: {prompt_type}")
    else:
        layers_to_steer = list(range(model.config.num_hidden_layers))
        logger.info(f"Using all {model.config.num_hidden_layers} layers for steering with prompt type: {prompt_type}")

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()

    # Run steered generation for each layer
    for steering_layer in layers_to_steer:
        steering_vec = bs_act_vector[steering_layer]
        
        # Lists to store generations for this layer
        thinking_steered, final_steered, total_steered = [], [], []

        logger.info(f"Running inference for layer {steering_layer}...")
        pbar = tqdm(total=len(user_prompts_df), desc=f"Layer {steering_layer} steered inference")
        for _, row in user_prompts_df.iterrows():
            user_prompt = str(row[prompt_column])
            
            generations = steered_single_message_inference(
                lm=model, 
                tok=tokenizer, 
                system_prompt=selected_system_prompt, 
                user_prompt=user_prompt, 
                steering_vec=steering_vec,
                layer_idx=steering_layer,
                max_new_tokens=max_new_tokens,
                alpha=alpha
            )
            
            thinking_steered.append(generations["thinking"])
            final_steered.append(generations["final"])
            total_steered.append(generations["total"])
            
            pbar.update(1)
        pbar.close()

        # Add steered generations to dataframe for this layer
        user_prompts_df[f"thinking_steered_{prompt_type}_layer_{steering_layer}"] = thinking_steered
        user_prompts_df[f"final_steered_{prompt_type}_layer_{steering_layer}"] = final_steered
        user_prompts_df[f"total_steered_{prompt_type}_layer_{steering_layer}"] = total_steered

    # Save generations (all layers in one file)
    if layer_idx is not None:
        output_csv_path = Path(output_dir, f"steered_generations_{prompt_type}_layer_{layer_idx}_{timestamp}.csv")
    else:
        output_csv_path = Path(output_dir, f"steered_generations_{prompt_type}_all_layers_{timestamp}.csv")
    
    user_prompts_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved steered generations to {output_csv_path}")

    logger.info("Steered generation completed!")

if __name__ == '__main__':
    Fire(main)