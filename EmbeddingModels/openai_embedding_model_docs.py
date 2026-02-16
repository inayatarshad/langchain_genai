from langchain import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding= OpenAIEmbeddings(model= "text-embedding-3-small", dimensions=32)

documents= [
    "Indias capital is New Delhi",
    "France capital is Paris",
    "Germany capital is Berlin",
    "Italy capital is Rome",]

result= embedding.embed_query("What is the capital of India?")
print(str(result))
