from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline
from dotenv import load_dotenv


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
messages = [
    SystemMessage(content="You are a loving mother"),
    HumanMessage(content="Am i lovable or not🥲?")
]
result = model.invoke(messages)
messages.append(AIMessage(content = result.content))
print(messages)
