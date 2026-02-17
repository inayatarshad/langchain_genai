from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st

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

st.header("Chatgpt-like bot hehe")
#chat_history = []
user_input = st.text_input("You: ")
send_button = st.button("Send")
if send_button and user_input:
    #chat_history.append(user_input)
    result=model.invoke(user_input)
    #chat_history.append(result.content)
    st.write(f"Bot: {result.content}")
if user_input.lower() == "exit":
        st.write("Goodbye!")
        st.stop()
