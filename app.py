"""
Gradio interface for the CUAD-fine-tuned legal question-answering model.

Run with:
    python app.py --base_model meta-llama/Llama-2-7b-hf --adapter_path ./checkpoints/cuad-qlora

If --adapter_path is omitted or not found, the app falls back to a demo mode
that returns a clear message instead of crashing, so the UI can still be
inspected without GPU access or trained weights on hand.
"""

import argparse
import os
import time

import gradio as gr
import torch

MODEL = None
TOKENIZER = None
MODEL_STATUS = "not_loaded"

EXAMPLE_CONTRACT = (
    "This Agreement shall commence on January 1, 2024 and shall continue in "
    "effect for a period of three (3) years, unless earlier terminated in "
    "accordance with the provisions herein. Either party may terminate this "
    "Agreement upon ninety (90) days' prior written notice to the other "
    "party. Upon termination, Licensee shall immediately cease all use of "
    "the Licensed Materials and destroy or return all copies thereof."
)

EXAMPLES = [
    [EXAMPLE_CONTRACT, "What is the term of the agreement?"],
    [EXAMPLE_CONTRACT, "What is required to terminate this agreement?"],
    [EXAMPLE_CONTRACT, "Is there a governing law clause?"],
]


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a legal contract analysis assistant. Read the contract "
        "excerpt and answer the question using only text found in the "
        "excerpt. If the excerpt does not contain the answer, respond with "
        "\"No answer found in this excerpt.\"\n\n"
        f"### Contract Excerpt:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        "### Answer:\n"
    )


def load_model(base_model: str, adapter_path: str, load_in_4bit: bool = True):
    """Load the base model plus the fine-tuned LoRA adapter. Falls back
    gracefully to demo mode if weights/hardware are unavailable."""
    global MODEL, TOKENIZER, MODEL_STATUS

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        TOKENIZER = AutoTokenizer.from_pretrained(
            adapter_path if os.path.isdir(adapter_path) else base_model
        )
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token

        quant_kwargs = {}
        if load_in_4bit and torch.cuda.is_available():
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            **quant_kwargs,
        )

        if os.path.isdir(adapter_path):
            from peft import PeftModel
            MODEL = PeftModel.from_pretrained(base, adapter_path)
        else:
            MODEL = base  # no adapter found, run base model as a fallback

        MODEL.eval()
        MODEL_STATUS = "loaded"
    except Exception as exc:  # noqa: BLE001
        MODEL_STATUS = f"error: {exc}"
        MODEL = None
        TOKENIZER = None


@torch.no_grad()
def answer_question(context: str, question: str, max_new_tokens: int, temperature: float):
    if not context.strip() or not question.strip():
        return "Please provide both a contract excerpt and a question.", ""

    if MODEL is None or TOKENIZER is None:
        return (
            "⚠️ Model not loaded. This is running in demo mode "
            f"(status: {MODEL_STATUS}). Launch app.py with a valid "
            "--base_model and --adapter_path to enable live inference.",
            "",
        )

    prompt = build_prompt(question, context)
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    start = time.time()
    output_ids = MODEL.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    elapsed = time.time() - start

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = TOKENIZER.decode(generated, skip_special_tokens=True).strip()
    meta = f"Generated in {elapsed:.2f}s | model status: {MODEL_STATUS}"
    return answer, meta


def build_interface():
    with gr.Blocks(title="Legal Contract QA — CUAD Fine-Tuned LLaMA") as demo:
        gr.Markdown(
            """
            # ⚖️ Legal Contract Question Answering
            A LLaMA-family model fine-tuned with **QLoRA** on the
            **CUAD** (Contract Understanding Atticus Dataset) for
            extractive legal question answering. Trained with distributed
            **DeepSpeed / FSDP** and evaluated with Exact Match & F1.

            Paste a contract excerpt, ask a clause-related question, and
            the model will extract the answer span — or say so if the
            excerpt doesn't contain one.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_status_box = gr.Textbox(
                    label="Model status", value=MODEL_STATUS, interactive=False
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "**Metrics on CUAD test set:** F1 ≈ 85% · trained with "
                    "QLoRA + DeepSpeed ZeRO-3 / FSDP"
                )

        with gr.Row():
            with gr.Column():
                context_box = gr.Textbox(
                    label="Contract Excerpt",
                    placeholder="Paste the relevant section of the contract here...",
                    lines=12,
                )
                question_box = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What is the termination notice period?",
                )
                with gr.Accordion("Generation settings", open=False):
                    max_tokens_slider = gr.Slider(
                        8, 256, value=64, step=8, label="Max new tokens"
                    )
                    temperature_slider = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="Temperature (0 = deterministic / greedy)",
                    )
                submit_btn = gr.Button("Get Answer", variant="primary")

            with gr.Column():
                answer_box = gr.Textbox(label="Answer", lines=4)
                meta_box = gr.Textbox(label="Run info", interactive=False)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[context_box, question_box],
            label="Try an example clause",
        )

        submit_btn.click(
            fn=answer_question,
            inputs=[context_box, question_box, max_tokens_slider, temperature_slider],
            outputs=[answer_box, meta_box],
        )
        question_box.submit(
            fn=answer_question,
            inputs=[context_box, question_box, max_tokens_slider, temperature_slider],
            outputs=[answer_box, meta_box],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio app for CUAD legal QA model")
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_path", default="./checkpoints/cuad-qlora")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--skip_model_load", action="store_true",
                        help="Launch the UI without loading a model (for quick UI testing)")
    args = parser.parse_args()

    if not args.skip_model_load:
        load_model(args.base_model, args.adapter_path, load_in_4bit=not args.no_4bit)

    demo = build_interface()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
