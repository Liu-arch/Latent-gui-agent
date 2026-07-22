#!/usr/bin/env python3
"""Load a local Qwen3-VL checkpoint and run one visual generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen3_gui_agent.lara_style_qwen3vl_agent import LaRAStyleQwen3VLAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--agent-wrapper", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model}")
    if not torch.cuda.is_available():
        raise RuntimeError("The PPU CUDA-compatible runtime is unavailable")

    image = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(image).rectangle((56, 56, 168, 168), fill="red")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "What color is the square? Answer with one word.",
                },
            ],
        }
    ]

    started = time.perf_counter()
    if args.agent_wrapper:
        agent = LaRAStyleQwen3VLAgent.from_pretrained(
            str(args.model),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=args.attn_implementation,
            local_files_only=True,
        )
        processor = agent.processor
        model = agent.model
    else:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    load_seconds = time.perf_counter() - started

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)

    generated_at = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    generation_seconds = time.perf_counter() - generated_at
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print(
        json.dumps(
            {
                "status": "PASS",
                "model": str(args.model),
                "model_class": type(model).__name__,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "visible_devices": torch.cuda.device_count(),
                "device_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "device_map": getattr(model, "hf_device_map", None),
                "attn_implementation": args.attn_implementation,
                "agent_wrapper": args.agent_wrapper,
                "input_token_count": int(inputs.input_ids.shape[1]),
                "pixel_values_shape": list(inputs.pixel_values.shape),
                "response": response,
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
