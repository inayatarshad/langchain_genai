from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from streamlit import button, text_input, write
from transformers import pipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
#import streamlit as st

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

#st.header("Chatgpt-like bot hehe")
chat_history = [
 SystemMessage(content="You are a helpful teacher")     
]
user_input = text_input(HumanMessage(content="You: "))
send_button = button("Send")
if send_button and user_input:
    #chat_history.append(user_input)
    result=model.invoke([HumanMessage(content=user_input)])
    #chat_history.append(result.content)
    chat_history.append(AIMessage(content=f"Bot: {result.content}"))
if user_input.lower() == "exit":
        print("Goodbye!")
print(chat_history)