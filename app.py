"""Gradio chat interface for the SmolLM3-3B model."""

import argparse
import sys
import traceback

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

tokenizer = None
model = None
device = "cpu"


def detect_device():
    """Return the best available torch device name."""
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def load_model():
    """Load the tokenizer and model onto the best available device."""
    global tokenizer, model, device

    device = detect_device()

    print("🚀 Loading SmolLM3-3B model...")
    print(f"📱 Device: {device}")
    print(f"🔧 PyTorch version: {torch.__version__}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model_kwargs = {"torch_dtype": torch.float16 if device != "cpu" else torch.float32}
        if device == "cuda":
            # accelerate shards the model across the available GPUs.
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

        if device != "cuda":
            model = model.to(device)

        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:  # noqa: BLE001 - startup failure must be reported to the user
        print(f"❌ Error loading model: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print(f"✅ Model loaded successfully on {device}")


def format_prompt(prompt, enable_thinking=False):
    """Build chat prompt with tokenizer template when available."""
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older tokenizers do not accept the enable_thinking flag.
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return f"User: {prompt}\nAssistant:"


def chat(prompt, enable_thinking=False, max_tokens=256, temperature=0.6, top_p=0.95):
    """Generate a response using SmolLM3-3B."""
    if model is None or tokenizer is None:
        return "Model is not loaded yet. Please restart the application."

    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    try:
        text = format_prompt(prompt, enable_thinking=enable_thinking)

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        temperature = float(temperature)
        generation_kwargs = {
            "max_new_tokens": max(1, int(max_tokens)),
            "pad_token_id": tokenizer.pad_token_id,
        }

        if temperature > 0:
            generation_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=float(top_p),
            )
        else:
            generation_kwargs["do_sample"] = False

        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)

        output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001 - surface generation errors in the UI
        traceback.print_exc()
        return f"Error generating response: {exc}"


def create_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="SmolLM3-3B Chatbot", theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            """
            # 🤖 SmolLM3-3B Chatbot

            A local AI chatbot powered by SmolLM3-3B. This model runs entirely on your machine!

            **Features:**
            - 💬 Natural conversation
            - 🧠 Extended thinking mode for reasoning
            - ⚡ GPU acceleration (if available)
            - 🔒 Complete privacy (no data sent to external servers)
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask me anything...",
                    lines=3,
                    max_lines=10,
                )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

            with gr.Column(scale=1):
                thinking_mode = gr.Checkbox(
                    label="🧠 Extended Thinking Mode",
                    value=False,
                    info="Enable reasoning traces",
                )

                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=256,
                    step=50,
                    label="Max Tokens",
                )

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.6,
                    step=0.1,
                    label="Temperature",
                )

                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p",
                )

        response_output = gr.Textbox(
            label="SmolLM3 Response",
            lines=10,
            max_lines=20,
            interactive=False,
        )

        inputs = [prompt_input, thinking_mode, max_tokens, temperature, top_p]

        submit_btn.click(fn=chat, inputs=inputs, outputs=response_output)
        prompt_input.submit(fn=chat, inputs=inputs, outputs=response_output)
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[prompt_input, response_output],
        )

        gr.Markdown(
            f"""
            ---
            **System Info:**
            - Device: {device.upper()}
            - Model: {MODEL_NAME}
            - PyTorch: {torch.__version__}
            """
        )

    return iface


def main():
    parser = argparse.ArgumentParser(description="SmolLM3-3B Gradio Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")

    args = parser.parse_args()

    load_model()

    print(f"🌐 Starting Gradio interface on {args.host}:{args.port}")

    interface = create_interface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
