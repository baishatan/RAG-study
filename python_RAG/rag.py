# from ollama import chat, Message

# messages = [
#     Message(role="system", content="You are a helpful assistant."),
#     Message(role="user", content="Who won the world series in 2020?"),
#     Message(role="assistant", content="The Los Angeles Dodgers won the World Series in 2020."),
#     Message(role="user", content="Where was it played?"),
# ]

# response = chat(messages=messages, model="qwen2.5:0.5b")
# print(response)

import numpy as np
from ollama import embeddings, chat, Message

class Kb:
    def __init__(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        self.docs = self.split_content(content)
        self.embeds = self.encode(self.docs)

    @staticmethod
    def split_content(content, max_length=50):
        chunks = []
        for i in range(0, len(content), max_length):
            chunks.append(content[i:i+max_length])
        return chunks
    
    @staticmethod
    def encode(texts):
        embeds = []
        for text in texts:
            response = embeddings(model="nomic-embed-text", prompt=text)
            embeds.append(response["embedding"])
        return np.array(embeds)
    
    @staticmethod
    def similarity(e1, e2):
        dot_product = np.dot(e1, e2)
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        return dot_product / (norm_e1 * norm_e2)

    def search(self, text):
        max_similarity = 0
        max_similarity_index = 0
        e = self.encode([text])[0]
        for idx, te in enumerate(self.embeds):
            similarity = self.similarity(e, te)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = idx
        return self.docs[max_similarity_index]

class RAG:
    def __init__(self, model, kb: Kb):
        self.model = model
        self.kb = kb
        self.prompt_template = """
        请根据以下知识库回答问题：\n\n
        知识库：%s\n\n
        问题：%s\n\n
        """

    def chat(self, text):
        # 在知识库里面查找
        context = self.kb.search(text)
        # 将context拼接到prompt
        prompt = self.prompt_template % (context, text)
        response = chat(self.model, [Message(role='system',content=prompt)])
        return response['message']




kb = Kb('爱因斯坦.txt')
rag = RAG("qwen2.5:0.5b", kb)
while True: 
    q = input("请输入问题：")
    r = rag.chat(q)
    print('Assistant:', r['content'])
