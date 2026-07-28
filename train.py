"""
Distributed QLoRA fine-tuning of a LLaMA-family model on CUAD legal QA.

Supports two distributed backends, chosen at launch time (not in code):
  * DeepSpeed ZeRO-3 (offload)  -> `deepspeed src/train.py --deepspeed configs/ds_zero3_config.json`
  * FSDP via Accelerate         -> `accelerate launch --config_file configs/fsdp_config.yaml src/train.py`

QLoRA (Dettmers et al., 2023) is applied by loading the base model in 4-bit
NF4 precision via bitsandbytes and training only low-rank adapters (PEFT),
which keeps memory usage low enough to fine-tune 7B+ parameter models on
consumer/mid-tier GPUs while DeepSpeed/FSDP shard optimizer state and
gradients across ranks for further scaling.
"""

import argparse
import logging
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from data_preprocessing import ChunkingConfig, build_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning on CUAD")
    p.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf",
                    help="Base model repo. Requires HF access + login (huggingface-cli login "
                        "or HF_TOKEN env var) since this repo is gated")
    p.add_argument("--train_json", default="cuda.json",
                    help="Path to the CUAD-format training JSON (default: cuda.json)")
    p.add_argument("--val_json", default=None,
                    help="Optional separate validation JSON; if omitted, a split is taken from --train_json")
    p.add_argument("--processed_dataset_dir", default=None,
                    help="If set, load a pre-tokenized dataset instead of re-processing")
    p.add_argument("--output_dir", default="./checkpoints/cuad-qlora")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--deepspeed", default=None, help="Path to DeepSpeed JSON config")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", default="none")
    return p.parse_args()


class LossConvergenceCallback(TrainerCallback):
    """Tracks train/eval loss history for post-hoc convergence monitoring
    and flags divergence (loss increasing over a sustained window) early."""

    def __init__(self, window: int = 5, divergence_factor: float = 1.5):
        self.history = []
        self.window = window
        self.divergence_factor = divergence_factor

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.history.append(("train", state.global_step, logs["loss"]))
        if "eval_loss" in logs:
            self.history.append(("eval", state.global_step, logs["eval_loss"]))
            self._check_divergence()

    def _check_divergence(self):
        eval_losses = [v for k, _, v in self.history if k == "eval"]
        if len(eval_losses) < self.window + 1:
            return
        recent = eval_losses[-self.window:]
        if recent[-1] > recent[0] * self.divergence_factor:
            logger.warning(
                "Validation loss rose from %.4f to %.4f over the last %d evals "
                "— possible divergence, consider lowering the learning rate.",
                recent[0], recent[-1], self.window,
            )


def load_quantized_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.processed_dataset_dir:
        dataset = load_from_disk(args.processed_dataset_dir)
    else:
        cfg = ChunkingConfig(max_seq_length=args.max_seq_length)
        dataset = build_dataset(args.train_json, args.val_json, tokenizer, cfg)

    model = load_quantized_model(args.model_name)
    model = apply_lora(model, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    convergence_cb = LossConvergenceCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=[convergence_cb],
    )

    logger.info("Starting distributed QLoRA fine-tuning...")
    trainer.train()

    logger.info("Saving final LoRA adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Persist loss history for later plotting / reporting.
    import json
    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump(convergence_cb.history, f, indent=2)


if __name__ == "__main__":
    main()
