from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2"
)

#response = model.invoke("tell me something about the sun in 2 sentences")
#print(response.content)