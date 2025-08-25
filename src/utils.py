import torch as t
from typing import Tuple

def parser(completion_token_ids: t.Tensor) -> Tuple[t.Tensor, t.Tensor | None]:
    all_token_idx = t.arange(len(completion_token_ids), device="cpu") 

    start_thinking = t.tensor([200005, 35644, 200008], device="cpu") # <|channel|>analysis<|message|>
    stop_thinking = 200007 # <|end|>
    start_final = t.tensor([200006, 173781, 200005, 17196, 200008], device="cpu") # <|start|>assistant<|channel|>final<|message|>
    stop_final = 200002 # <|return|>

    if not t.equal(completion_token_ids[: len(start_thinking)], start_thinking):
        raise ValueError("Expect sequence to always start with '<|channel|>analysis<|message|>'")

    start_thinking_idx = len(start_thinking)
    remainder = completion_token_ids[start_thinking_idx:]
    stop_thinking_matches = (remainder == stop_thinking).nonzero(as_tuple=True)[0]
    if stop_thinking_matches.numel() == 0:
        # Cut off during thinking
        thinking_token_idxs = all_token_idx[start_thinking_idx:]
        final_token_idxs = None

        return thinking_token_idxs, final_token_idxs
    stop_thinking_idx = int(stop_thinking_matches[0].item())
    thinking_token_idxs = all_token_idx[start_thinking_idx:start_thinking_idx+stop_thinking_idx]

    # After stop_thinking, expect immediate start_final
    after_thinking = remainder[stop_thinking_idx+1:]
    if after_thinking.numel() < len(start_final) or not t.equal(after_thinking[: len(start_final)], start_final):
        # No final present
        final_token_idxs = None

        return thinking_token_idxs, final_token_idxs

    final_body = after_thinking[len(start_final):]
    stop_final_matches = (final_body == stop_final).nonzero(as_tuple=True)[0]
    final_start_idx = start_thinking_idx + stop_thinking_idx + 1 + len(start_final)
    if stop_final_matches.numel() == 0:
        # Cut off during final
        final_token_idxs = all_token_idx[final_start_idx:]
    else:
        final_stop_idx = int(stop_final_matches[0].item())
        final_token_idxs = all_token_idx[final_start_idx:final_start_idx+final_stop_idx]

    return thinking_token_idxs, final_token_idxs