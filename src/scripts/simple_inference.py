# FROM: https://huggingface.co/blog/welcome-openai-gpt-oss#local-inference

import logging
from typing import Literal

from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    model_id: Literal["openai/gpt-oss-20b", "openai/gpt-oss-120b"] = "openai/gpt-oss-20b",
    system_prompt_path: str = "src/data/system_prompt.txt",
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What does the flag --deep-history do in git clone?"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    generated_ids = model.generate(**inputs, max_new_tokens=2096)

    print(tokenizer.decode(generated_ids[0][prompt_len:]))


if __name__ == '__main__':
    Fire(main)
