"""
Data preprocessing for the CUAD (Contract Understanding Atticus Dataset)
legal question-answering task.

CUAD is distributed in SQuAD 2.0 style JSON: each example has a legal
contract (context), a clause-type question, and an answer span (or is
unanswerable). Legal contracts routinely exceed the model's max sequence
length, so this module implements a sliding-window chunking strategy that
preserves the answer span alignment across chunks, mirroring the standard
SQuAD/BERT long-context preprocessing approach but adapted for causal LMs
used in an instruction-tuned QA format.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    max_seq_length: int = 2048
    doc_stride: int = 256          # overlap between consecutive chunks
    max_question_length: int = 96
    keep_no_answer_ratio: float = 0.3  # fraction of unanswerable chunks kept


def load_cuad_raw(json_path: str) -> List[Dict]:
    """Load the raw CUAD SQuAD-style JSON file into a flat list of examples."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)["data"]

    examples = []
    for entry in raw:
        title = entry.get("title", "")
        for para in entry["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers", [])
                examples.append(
                    {
                        "id": qa["id"],
                        "title": title,
                        "question": qa["question"],
                        "context": context,
                        "answers": {
                            "text": [a["text"] for a in answers],
                            "answer_start": [a["answer_start"] for a in answers],
                        },
                        "is_impossible": qa.get("is_impossible", len(answers) == 0),
                    }
                )
    logger.info("Loaded %d QA examples from %s", len(examples), json_path)
    return examples


def _chunk_context(context: str, tokenizer, max_ctx_tokens: int, doc_stride: int):
    """Split a long context into overlapping token-based chunks, returning
    each chunk's text plus its character offset in the original context so
    that answer spans can be re-aligned."""
    tokens = tokenizer(
        context,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = tokens["offset_mapping"]
    input_ids = tokens["input_ids"]

    chunks = []
    start = 0
    n = len(input_ids)
    while start < n:
        end = min(start + max_ctx_tokens, n)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append(
            {
                "text": context[char_start:char_end],
                "char_offset": char_start,
            }
        )
        if end == n:
            break
        start = end - doc_stride
    return chunks


def build_prompt(question: str, context_chunk: str) -> str:
    """Instruction-style prompt used for causal-LM fine-tuning on extractive
    legal QA. Keeping the format terse and consistent helps the LoRA
    adapters converge quickly on a narrow domain like CUAD."""
    return (
        "You are a legal contract analysis assistant. Read the contract "
        "excerpt and answer the question using only text found in the "
        "excerpt. If the excerpt does not contain the answer, respond with "
        "\"No answer found in this excerpt.\"\n\n"
        f"### Contract Excerpt:\n{context_chunk}\n\n"
        f"### Question:\n{question}\n\n"
        "### Answer:\n"
    )


def preprocess_examples(
    examples: List[Dict],
    tokenizer,
    cfg: ChunkingConfig,
) -> List[Dict]:
    """Turn raw CUAD examples into chunked (prompt, target) training pairs."""
    processed = []
    reserved_for_qa = cfg.max_question_length + 256  # prompt scaffolding + answer
    max_ctx_tokens = cfg.max_seq_length - reserved_for_qa

    for ex in examples:
        context = ex["context"]
        question = ex["question"]
        answers = ex["answers"]
        has_answer = len(answers["text"]) > 0

        chunks = _chunk_context(context, tokenizer, max_ctx_tokens, cfg.doc_stride)

        kept_no_answer = 0
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_start = chunk["char_offset"]
            chunk_end = chunk_start + len(chunk_text)

            answer_in_chunk = None
            if has_answer:
                for text, astart in zip(answers["text"], answers["answer_start"]):
                    aend = astart + len(text)
                    if astart >= chunk_start and aend <= chunk_end:
                        answer_in_chunk = text
                        break

            if answer_in_chunk is None:
                # Down-sample unanswerable chunks so the model isn't
                # overwhelmed by "no answer" targets from long contracts.
                kept_no_answer += 1
                if kept_no_answer > max(1, int(len(chunks) * cfg.keep_no_answer_ratio)):
                    continue
                target = "No answer found in this excerpt."
            else:
                target = answer_in_chunk

            prompt = build_prompt(question, chunk_text)
            processed.append(
                {
                    "id": ex["id"],
                    "prompt": prompt,
                    "target": target,
                    "has_answer": answer_in_chunk is not None,
                }
            )

    logger.info("Expanded %d raw examples into %d chunked training pairs",
                len(examples), len(processed))
    return processed


def tokenize_for_causal_lm(batch: Dict, tokenizer, max_seq_length: int) -> Dict:
    """Concatenate prompt + target, mask prompt tokens from the loss so the
    model is only trained to generate the answer span."""
    full_texts = [p + t + tokenizer.eos_token for p, t in zip(batch["prompt"], batch["target"])]
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    labels = []
    for i, prompt in enumerate(batch["prompt"]):
        prompt_len = len(
            tokenizer(prompt, truncation=True, max_length=max_seq_length)["input_ids"]
        )
        ids = tokenized["input_ids"][i]
        label_row = list(ids)
        for j in range(min(prompt_len, len(label_row))):
            label_row[j] = -100
        # mask padding
        pad_id = tokenizer.pad_token_id
        label_row = [-100 if tok == pad_id else lab for tok, lab in zip(ids, label_row)]
        labels.append(label_row)

    tokenized["labels"] = labels
    return tokenized


def build_dataset(
    train_json: str,
    val_json: Optional[str],
    tokenizer,
    cfg: ChunkingConfig,
) -> DatasetDict:
    train_raw = load_cuad_raw(train_json)
    train_pairs = preprocess_examples(train_raw, tokenizer, cfg)
    train_ds = Dataset.from_list(train_pairs)

    splits = {"train": train_ds}

    if val_json:
        val_raw = load_cuad_raw(val_json)
        val_pairs = preprocess_examples(val_raw, tokenizer, cfg)
        splits["validation"] = Dataset.from_list(val_pairs)
    else:
        split = train_ds.train_test_split(test_size=0.05, seed=42)
        splits = {"train": split["train"], "validation": split["test"]}

    dsd = DatasetDict(splits)
    dsd = dsd.map(
        lambda batch: tokenize_for_causal_lm(batch, tokenizer, cfg.max_seq_length),
        batched=True,
        remove_columns=[c for c in dsd["train"].column_names if c not in ("id",)],
        desc="Tokenizing CUAD chunks",
    )
    return dsd


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Preprocess CUAD for fine-tuning")
    parser.add_argument("--train_json", default="cuda.json",
                        help="Path to the CUAD-format training JSON (default: cuda.json)")
    parser.add_argument("--val_json", default=None,
                        help="Optional separate validation JSON; if omitted, a split is taken from --train_json")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", default="./data/processed")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = ChunkingConfig(max_seq_length=args.max_seq_length)
    dataset = build_dataset(args.train_json, args.val_json, tok, cfg)
    dataset.save_to_disk(args.output_dir)
    logger.info("Saved processed dataset to %s", args.output_dir)
