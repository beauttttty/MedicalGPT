#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the fixed Day 2 smoke evaluation and save structured predictions.

Run this script from the MedicalGPT project root, for example:

CUDA_VISIBLE_DEVICES=1 python notes/day2/eval/run_smoke_eval.py \
  --base_model Qwen/Qwen3.5-0.8B \
  --system_prompt "你是一个医疗助手。" \
  --temperature 0 \
  --max_new_tokens 256

Each JSONL input record is evaluated as an independent single-turn
conversation. Only its ``prompt`` field is sent to the model, so
``expected_behavior`` is never leaked into the prompt.
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_INPUT_FILE = "./notes/day2/eval/smoke_prompts.jsonl"
DEFAULT_OUTPUT_FILE = "./notes/day2/e0_base/smoke_predictions.jsonl"
REQUIRED_FIELDS = {"id", "category", "prompt", "expected_behavior"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent, deterministic smoke-test inference."
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Hugging Face model ID or local base-model directory.",
    )
    parser.add_argument(
        "--lora_model",
        default="",
        help="Optional Hugging Face ID or local PEFT/LoRA adapter directory.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="",
        help="Optional tokenizer ID/directory; defaults to --base_model.",
    )
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--system_prompt",
        default="你是一个医疗助手。",
        help="The exact same system prompt must be used for every compared model.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.temperature < 0:
        parser.error("--temperature must be greater than or equal to 0")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be greater than 0")
    if args.eval_batch_size <= 0:
        parser.error("--eval_batch_size must be greater than 0")
    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty must be greater than 0")

    return args


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """Load and validate one JSON object per line."""
    cases: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            missing_fields = REQUIRED_FIELDS - set(case)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Missing fields at {path}:{line_number}: {missing}"
                )

            for field in REQUIRED_FIELDS:
                if not isinstance(case[field], str) or not case[field].strip():
                    raise ValueError(
                        f"Field '{field}' must be a non-empty string at "
                        f"{path}:{line_number}"
                    )

            cases.append(case)

    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")

    duplicate_ids = [
        case_id
        for case_id, count in Counter(case["id"] for case in cases).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate case IDs: {', '.join(sorted(duplicate_ids))}")

    return cases


def build_prompt(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    """Apply the tokenizer's own chat template to one independent case."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    generation_kwargs: Dict[str, Any],
) -> List[str]:
    """Generate one answer for every independently formatted prompt."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    outputs = model.generate(**inputs, **generation_kwargs)
    padded_prompt_length = inputs["input_ids"].shape[1]

    predictions = []
    for output_ids in outputs:
        generated_ids = output_ids[padded_prompt_length:]
        prediction = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        predictions.append(prediction)

    return predictions


def write_json(path: Path, value: Dict[str, Any]) -> None:
    """Write formatted JSON through a temporary file, then replace atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_cases(input_path)
    set_seed(args.seed)

    tokenizer_path = args.tokenizer_path or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    if args.lora_model:
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_model,
            device_map="auto",
        )
    else:
        model = base_model
    model.eval()

    do_sample = args.temperature > 0
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if do_sample:
        generation_kwargs["temperature"] = args.temperature

    saved_generation_config = {
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "do_sample": do_sample,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
    }
    run_config = {
        "base_model": args.base_model,
        "lora_model": args.lora_model or None,
        "tokenizer_path": tokenizer_path,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "generation_config": saved_generation_config,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(output_path.parent / "run_config.json", run_config)

    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    written_count = 0

    try:
        with temporary_output.open("w", encoding="utf-8") as output_file:
            for start in tqdm(
                range(0, len(cases), args.eval_batch_size),
                desc="Smoke evaluation",
            ):
                batch_cases = cases[start : start + args.eval_batch_size]
                formatted_prompts = [
                    build_prompt(tokenizer, case["prompt"], args.system_prompt)
                    for case in batch_cases
                ]
                predictions = generate_batch(
                    model,
                    tokenizer,
                    formatted_prompts,
                    generation_kwargs,
                )

                for case, prediction in zip(batch_cases, predictions):
                    result = {
                        "id": case["id"],
                        "category": case["category"],
                        "prompt": case["prompt"],
                        "expected_behavior": case["expected_behavior"],
                        "prediction": prediction,
                        "base_model": args.base_model,
                        "lora_model": args.lora_model or None,
                        "generation_config": saved_generation_config,
                    }
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write("\n")
                    output_file.flush()
                    written_count += 1

        os.replace(temporary_output, output_path)
    except BaseException:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(f"Saved {written_count} predictions to {output_path}")
    print(f"Saved run configuration to {output_path.parent / 'run_config.json'}")


if __name__ == "__main__":
    main()
