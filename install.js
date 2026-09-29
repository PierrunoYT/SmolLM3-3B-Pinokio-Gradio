module.exports = {
  run: [
    // Install the platform specific pytorch build first so that
    // requirements.txt (accelerate depends on torch) reuses it instead of
    // pulling the generic PyPI build.
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env"
        }
      }
    },
    // Install the python dependencies into the "env" virtual environment
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "input",
      params: {
        title: "Install Complete!",
        description: "Click 'Start' to launch the SmolLM3-3B chatbot. The model (~6GB) is downloaded on first launch."
      }
    }
  ]
}
