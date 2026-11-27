from deepeval.models import OllamaModel

# A plain string is treated as an OpenAI model name, so wrap it to use local Ollama.
eval_model = OllamaModel(
    model="qwen3:4b-instruct",
    base_url="http://localhost:11434",
    temperature=0,
)
