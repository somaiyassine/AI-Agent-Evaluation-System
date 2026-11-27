from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3:4b-instruct"
)

#response = model.invoke("tell me something about the sun in 2 sentences")
#print(response.content)