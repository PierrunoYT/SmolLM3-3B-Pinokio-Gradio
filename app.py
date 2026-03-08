import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import argparse
import sys

# Model configuration
model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"🚀 Loading SmolLM3-3B model...")
print(f"📱 Device: {device}")
print(f"🔧 PyTorch version: {torch.__version__}")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device in {"cuda", "mps"} else torch.float32,
            device_map="auto" if device in {"cuda", "mps"} else None
        )
    if device == "cpu":
        model = model.to(device)

    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


def format_prompt(prompt, enable_thinking=False):
    """Build chat prompt with tokenizer template when available."""
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if "enable_thinking" in tokenizer.apply_chat_template.__code__.co_varnames:
            template_kwargs["enable_thinking"] = enable_thinking

        return tokenizer.apply_chat_template(messages, **template_kwargs)

    return f"User: {prompt}\nAssistant:"

def chat(prompt, enable_thinking=False, max_tokens=256, temperature=0.6, top_p=0.95):
    """Generate response using SmolLM3-3B"""
    if not prompt.strip():
        return "Please enter a prompt."
    
    try:
        # Apply chat template
        text = format_prompt(prompt, enable_thinking=enable_thinking)
        
        # Tokenize input
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
            )
        
        # Decode response
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def create_interface():
    """Create and configure Gradio interface"""
    
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
                    max_lines=10
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)
            
            with gr.Column(scale=1):
                thinking_mode = gr.Checkbox(
                    label="🧠 Extended Thinking Mode",
                    value=False,
                    info="Enable reasoning traces"
                )
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=256,
                    step=50,
                    label="Max Tokens"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.6,
                    step=0.1,
                    label="Temperature"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p"
                )
        
        response_output = gr.Textbox(
            label="SmolLM3 Response",
            lines=10,
            max_lines=20,
            interactive=False
        )
        
        # Event handlers
        submit_btn.click(
            fn=chat,
            inputs=[prompt_input, thinking_mode, max_tokens, temperature, top_p],
            outputs=response_output
        )
        
        prompt_input.submit(
            fn=chat,
            inputs=[prompt_input, thinking_mode, max_tokens, temperature, top_p],
            outputs=response_output
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[prompt_input, response_output]
        )
        
        # System info
        gr.Markdown(
            f"""
            ---
            **System Info:**
            - Device: {device.upper()}
            - Model: {model_name}
            - PyTorch: {torch.__version__}
            """
        )
    
    return iface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmolLM3-3B Gradio Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    
    args = parser.parse_args()
    
    print(f"🌐 Starting Gradio interface on {args.host}:{args.port}")
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )
