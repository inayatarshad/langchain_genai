#with_structured_output is used so that we can have proper structured outputs
#from llms apart from text ansers bcz structured outputs like json formats are useful when we want to connect
#connect our llm with other databses or external tools, but as with_structured_output is not compatible with huggingface so we are using openai in this (just for practice we dont hv openai api)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotator

#typed dict is an ez way out when we want to specify that these variables are to kept as string or int or whatsoever so for next user its ez
model = ChatOpenAI()

class Review(TypedDict):
    summary = Annotator[str, "A brief summary of the review given"]
    sentiment = Annotator[str, "A final verdict of the entire review if it was positive or negative"]
#annotator is used for the purpose of keeping the prompt in a good detailed view rather than just saying summmary or sentiment
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""This is a chatbot model, often people try to do chatbots for illegal
                      activities that are not much supported this needs some action to be taken against, 
                      models should be fine tuned and held properly responsible, 
                      parents should keep a check on their children uncontrolled use of AI""") 

print(result)
print(result["Summary"])
print(result["Sentiment"])
