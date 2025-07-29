# SmolLM3-3B Pinokio Package

A complete Pinokio installation package for running SmolLM3-3B locally with a beautiful Gradio web interface.

⚠️ **This package is designed exclusively for Pinokio and should not be installed directly. Use Pinokio to install and manage this application.**

## 🚀 Features

- **🤖 SmolLM3-3B Model**: Advanced 3B parameter language model from HuggingFace
- **💬 Gradio Interface**: Clean, responsive web interface with soft theme
- **🧠 Extended Thinking Mode**: Enable reasoning traces for complex queries
- **⚡ GPU Acceleration**: Automatic CUDA detection with FP16 precision
- **🔒 Complete Privacy**: Runs entirely offline, no data sent externally
- **🌐 Cross-Platform**: Windows, macOS, and Linux support
- **⚙️ Customizable**: Adjustable temperature (0.1-2.0), top-p (0.1-1.0), and max tokens (50-1000)
- **🖥️ Command Line Support**: Configurable host, port, and sharing options

## 📋 Requirements

- **RAM**: 8GB+ (12GB+ recommended for smooth operation)
- **Storage**: ~6GB for model files (downloaded automatically)
- **GPU**: Optional but highly recommended (8GB+ VRAM for best performance)
- **OS**: Windows 10/11, macOS, or Linux

## 🛠️ Installation

### Via Pinokio (Only Supported Method)

1. Install [Pinokio](https://pinokio.computer/)
2. Open Pinokio and navigate to "Discover"
3. Search for "SmolLM3-3B" or paste this repository URL
4. Click "Install" and wait for the installation to complete
5. Click "Start" to launch the chatbot
6. Open the web interface when it becomes available

⚠️ **Note**: This package is specifically designed for Pinokio's package management system and automated dependency handling. Manual installation is not supported and may result in configuration issues.

## 🎯 Usage

1. **Start the Application**: Click "Start SmolLM3 Chatbot" in Pinokio
2. **Open Web Interface**: Click "Open SmolLM3 Chatbot" when available (defaults to http://127.0.0.1:7860)
3. **Chat**: Type your message and press Send or hit Enter
4. **Advanced Options**:
   - Toggle "🧠 Extended Thinking Mode" for reasoning traces
   - Adjust temperature (0.1-2.0) for response creativity
   - Modify top-p (0.1-1.0) for response diversity
   - Set max tokens (50-1000) for response length
   - Use "Clear" button to reset the conversation

### Command Line Options

The Python application supports several command-line arguments:
- `--port`: Port to run the server on (default: 7860)
- `--host`: Host to run the server on (default: 127.0.0.1)
- `--share`: Create a public Gradio link for sharing

## 🏗️ Project Structure

```
SmolLM3-3B/
├── pinokio.js              # Main Pinokio configuration
├── install.js              # Installation workflow
├── start.js                # Application startup
├── update.js               # Update workflow
├── reset.js                # Reset/cleanup workflow
├── torch.js                # PyTorch installation
├── app.py                  # Main Gradio application
├── requirements.txt        # Python dependencies specification
├── icon.png                # Project icon
├── README.md               # This file
├── .gitignore              # Git ignore rules
└── app/                    # Created during installation
    ├── env/                # Python virtual environment
    ├── app.py              # Main application (copied from root)
    └── requirements.txt    # Python dependencies (copied from root)
```

## 🔧 Technical Details

### Model Information
- **Model**: HuggingFaceTB/SmolLM3-3B
- **Parameters**: 3 billion
- **Context Length**: Up to 8192 tokens
- **License**: Apache 2.0
- **Precision**: FP16 on GPU, FP32 on CPU

### Dependencies
All Python dependencies are specified in `requirements.txt`:
- **PyTorch** (≥2.0.0) with CUDA support when available
- **Transformers** (≥4.40.0) for model loading and inference
- **Gradio** (≥4.0.0) for the web interface
- **Accelerate, Tokenizers, SafeTensors** for optimized model handling
- **SentencePiece, Protobuf** for text processing
- **NumPy, PyYAML, Requests** and other utility libraries

Note: PyTorch installation with appropriate CUDA support is handled automatically by the Pinokio installation scripts.

### GPU Support Matrix

| Platform | NVIDIA | AMD | CPU |
|----------|--------|-----|-----|
| Windows | CUDA 12.8 + XFormers | DirectML | CPU-only |
| Linux | CUDA 12.8 + XFormers + SageAttention | ROCm 6.2.4 | CPU-only |
| macOS | N/A | N/A | CPU + Metal |

## 🚨 Troubleshooting

### Common Issues

**Slow responses on CPU**
- This is normal - CPU inference is much slower than GPU
- Consider using a GPU for better performance
- Reduce max tokens to speed up generation

**Out of memory errors**
- Reduce max tokens in the interface
- Close other applications to free RAM
- Use CPU mode if GPU memory is insufficient

**Model download fails**
- Check internet connection
- Model files (~6GB) download automatically on first use
- Downloads are cached for future use

**Installation errors**
- Ensure you have sufficient disk space (~10GB)
- Check that Python 3.9+ is available
- Try running as administrator on Windows

### Performance Tips

- **GPU Users**: Enable GPU acceleration for 10-50x faster inference
- **CPU Users**: Use lower token limits and simpler prompts
- **Memory**: Close unnecessary applications during use
- **Thinking Mode**: Disable for faster responses when reasoning isn't needed

## 🔄 Updates

The package includes an automatic update system:

1. Click "Update" in the Pinokio interface
2. Wait for dependencies and model updates to complete
3. Restart the application to use the latest version

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork this repository
2. Make your changes to the Pinokio scripts or Gradio interface
3. Test thoroughly on your target platform
4. Submit a pull request with a clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The SmolLM3-3B model is licensed under Apache 2.0 by HuggingFace.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for the SmolLM3-3B model
- [Pinokio](https://pinokio.computer/) for the amazing AI package manager
- [Gradio](https://gradio.app/) for the web interface framework
- The open-source AI community for making this possible

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and community support
- **Documentation**: Check the [Pinokio documentation](https://docs.pinokio.computer/) for general Pinokio help

---

**Made with ❤️ for the AI community**