from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv 
import streamlit as st
import os

load_dotenv()

st.header("Research tool")

# Initialize model with caching and proper configuration
@st.cache_resource
def get_model():
    return HuggingFacePipeline.from_model_id(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        device=-1,  # Use CPU (-1), or 0 for GPU
        pipeline_kwargs={
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
        },
        model_kwargs={
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
        }
    )

with st.spinner("Loading model (this may take a minute on first run)..."):
    model = get_model()

query = st.text_input("Enter your query here:")

if st.button("Summarize"):
    if query:
        with st.spinner("Generating response..."):
            try:
                result = model.invoke(query)
                st.success("Response:")
                st.write(result)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query first!")