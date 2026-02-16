from langchain import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding= OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
result= embedding.embed_query("What is the capital of india?")
print(str(result))
