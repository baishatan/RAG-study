from ollama import embeddings

embeddings = embeddings(model="nomic-embed-text",prompt="hello")
print(embeddings['embedding'])
print(len(embeddings['embedding']))
print(type(embeddings['embedding']))