# Gradio Demo

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Set up model inference server

3. Set environment variables

```bash
export BACKEND_HOST="***"  # the host of model inference server
export MODEL_NAME="***"  # the name of model
export API_KEY=="***"  # the api key of model, can be ignored if no auth
export CHAT_ENDPOINT="***"  # the endpoint of chat model, users can use different models for different tasks
export DOC_QA_ENDPOINT="***"  # the endpoint of doc-qa model, users can use different models for different tasks
export PLUGIN_ENDPOINT="***"  # the endpoint of plugin model, users can use different models for different tasks
export QA_GENERATOR_ENDPOINT="***"  # the endpoint of qa-generator model, users can use different models for different tasks
export EXTRACT_ENDPOINT="***"  # the endpoint of extract model, users can use different models for different tasks

# Plugins can be replaced according to actual conditions
export WEATHER_PLUGIN_URL="***"  # the url of weather plugin
export BCE_APP_CODE="***"  # the app code of bce
```

4. Run the demo

```bash
gradio app.py
```