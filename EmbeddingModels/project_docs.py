from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents=[
    "peso is currency of phillipines",
    "dollar is currency of USA",
    "euro is currency of europe",
    "rupee is currency of Pakistan",
]
documents_embeddings=embedding.embed_documents(documents)

query="what is currency of pakistan?"
query_embeddings=embedding.embed_query(query)

print(cosine_similarity([query_embeddings], documents_embeddings))
