from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from regex import template
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

# Step 1: Create the underlying HuggingFace pipeline
hf_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # or any model you prefer
    max_new_tokens=512
)

# Step 2: Wrap it in HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Step 3: Now wrap THAT in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

st.header("Research tool")

paper_input= st.selectbox("Select research paper name:", ["word2vec conversion", "TRUSTX UAV control", "BERT for semantic analysis", "Langchain for research"])
style_input= st.selectbox("Select output style:", ["Beginner-friendly summary", "Technical survey","Extremely rigorous","moderate"])

length_input= st.selectbox("Select output length:", ["Short (1-2 sentences)", "Medium (1-2 paragraphs)", "Long(1-2 pages)"])


template = load_prompt("prompt_template.json")



if st.button("Summarize"):
   chain = template | model
   result=chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})
   
   st.write(result.content)