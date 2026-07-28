"""
Evaluate a fine-tuned CUAD QA model using the standard SQuAD-style Exact
Match (EM) and token-level F1 metrics.
"""

import argparse
import collections
import json
import logging
import re
import string

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_preprocessing import build_prompt, load_cuad_raw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NO_ANSWER_PHRASE = "No answer found in this excerpt."


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace —
    the standard SQuAD normalization used before comparing EM/F1."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths):
    if not ground_truths:
        ground_truths = [NO_ANSWER_PHRASE]
    return max(metric_fn(gt, prediction) for gt in ground_truths)


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=2048).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, examples, max_context_chars: int = 6000):
    em_total, f1_total = 0.0, 0.0
    per_example = []

    for ex in tqdm(examples, desc="Evaluating"):
        context = ex["context"][:max_context_chars]
        prompt = build_prompt(ex["question"], context)
        prediction = generate_answer(model, tokenizer, prompt)

        ground_truths = ex["answers"]["text"]
        em = metric_max_over_ground_truths(compute_exact, prediction, ground_truths)
        f1 = metric_max_over_ground_truths(compute_f1, prediction, ground_truths)

        em_total += em
        f1_total += f1
        per_example.append({"id": ex["id"], "prediction": prediction,
                            "gold": ground_truths, "em": em, "f1": f1})

    n = max(len(examples), 1)
    return {"exact_match": 100.0 * em_total / n, "f1": 100.0 * f1_total / n}, per_example


def main():
    parser = argparse.ArgumentParser(description="Evaluate CUAD QA model (EM/F1)")
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_path", required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--output_file", default="./eval_results.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally evaluate on a subset for a quick check")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    examples = load_cuad_raw(args.test_json)
    if args.limit:
        examples = examples[: args.limit]

    scores, per_example = evaluate(model, tokenizer, examples)
    logger.info("Exact Match: %.2f | F1: %.2f", scores["exact_match"], scores["f1"])

    with open(args.output_file, "w") as f:
        json.dump({"scores": scores, "per_example": per_example}, f, indent=2)
    logger.info("Wrote detailed results to %s", args.output_file)


if __name__ == "__main__":
    main()

