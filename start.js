module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        message: [
          "python app.py --port {{port}}"
        ],
        on: [{
          // Capture the local URL Gradio prints once the server is up
          "event": "/(http:\/\/[0-9.:]+)/",
          "done": true
        }]
      }
    },
    {
      // Setting 'url' makes pinokio.js display the "Open Web UI" tab
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
