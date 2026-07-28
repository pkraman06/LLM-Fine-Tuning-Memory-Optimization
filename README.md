# Distributed LLM Fine-Tuning & Memory Optimization Pipeline

A distributed fine-tuning pipeline for LLaMA-family models, applying
**QLoRA** for memory-efficient training and **DeepSpeed ZeRO-3 / FSDP** for
multi-GPU scaling. Fine-tuned on **CUAD** (Contract Understanding Atticus
Dataset) for legal question answering, evaluated with **Exact Match (EM)**
and **F1**, and served through a **Gradio** interface.

## Project layout

```
llm-finetune-pipeline/
├── app.py                        # Gradio inference UI (token + model ID entered in-app)
├── requirements.txt
├── configs/
│   ├── ds_zero3_config.json      # DeepSpeed ZeRO-3 (CPU offload) config
│   └── fsdp_config.yaml          # Accelerate/FSDP config
├── scripts/
│   ├── launch_deepspeed.sh       # Multi-GPU launch via DeepSpeed
│   └── launch_fsdp.sh            # Multi-GPU launch via FSDP/Accelerate
└── src/
    ├── data_preprocessing.py     # CUAD loading + long-document chunking
    ├── train.py                  # QLoRA fine-tuning entrypoint
    └── evaluate.py               # EM/F1 evaluation
```

## Quickstart in Google Colab

```python
# 1. Clone your repo
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Upload cuda.json (Colab file browser, or the snippet below)
from google.colab import files
uploaded = files.upload()   # select your cuda.json

# 4. Launch the Gradio app (Colab needs a public link)
!python app.py --share
```

Click the `https://xxxxx.gradio.live` link Gradio prints. Inside the app:
1. Paste your **Hugging Face token** (from https://huggingface.co/settings/tokens — a "Read" token is enough; required because `meta-llama/Llama-2-7b-hf` is gated).
2. Enter the **Base Model ID** (defaults to `meta-llama/Llama-2-7b-hf`).
3. Enter a **LoRA Adapter** path/repo if you've already trained one, or leave it blank to just use the base model.
4. Click **Load Model**, wait for the ✅ status, then ask questions.

No command-line flags for the model are needed — everything is entered and loaded from inside the running app.

## Training (outside the Gradio app)

### 1. Data
Place your CUAD-format (SQuAD-style) JSON at the project root as
**`cuda.json`** — this is the default `--train_json` for `src/train.py`,
`src/data_preprocessing.py`, and both launch scripts. If you don't pass
`--val_json`, a 95/5 validation split is carved out of `cuda.json`
automatically. Long contracts are split into overlapping token-windowed
chunks so answer spans stay aligned even when a contract exceeds the
model's context length.

### 2. Authenticate with Hugging Face
`meta-llama/Llama-2-7b-hf` is gated — set your token once so all scripts
pick it up:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```
Both launch scripts automatically run `huggingface-cli login --token "$HF_TOKEN"` if `HF_TOKEN` is set.

### 3. Train (QLoRA + distributed backend)

**DeepSpeed ZeRO-3** (params/optimizer state sharded + CPU offload):
```bash
HF_TOKEN=hf_xxxx bash scripts/launch_deepspeed.sh
```

**FSDP** (via Accelerate, full-shard + CPU offload):
```bash
HF_TOKEN=hf_xxxx bash scripts/launch_fsdp.sh
```

Both wrap `src/train.py`: the base model loads in 4-bit NF4 via
`bitsandbytes` (QLoRA), LoRA adapters attach via `peft`
(`q/k/v/o_proj`, `gate/up/down_proj`), and only the adapters train —
keeping memory low while DeepSpeed/FSDP shard the surrounding state across
GPUs. Loss is logged every `--logging_steps`, a `LossConvergenceCallback`
flags sustained eval-loss increases, and history is saved to
`loss_history.json` in the output directory.

### 4. Evaluate (EM / F1)
```bash
python src/evaluate.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --adapter_path ./checkpoints/cuad-qlora \
  --test_json ./cuda.json
```
Reports corpus-level Exact Match and F1 (standard SQuAD normalization) and
writes per-example predictions to `eval_results.json`.

## Notes on the memory-optimization stack

| Technique | Role |
|---|---|
| QLoRA (4-bit NF4 + double quant) | Loads the frozen base model in 4-bit, trains small LoRA adapters in bf16 — cuts trainable-parameter memory by orders of magnitude |
| Gradient checkpointing | Trades compute for activation memory |
| DeepSpeed ZeRO-3 | Shards optimizer states, gradients, and parameters across GPUs, with optional CPU offload |
| FSDP (alternative) | Native PyTorch full-shard data parallelism, comparable scaling to ZeRO-3 |
| Sliding-window chunking | Keeps long legal contracts within the model's context window without losing answer alignment |
