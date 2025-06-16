import os
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import sys

# Load the API key from .env
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Setup MistralAI LLM with LangChain
llm = ChatMistralAI(api_key=api_key, model="mistral-small")

# Mental health support prompt
template = """
You are a friendly and empathetic mental health assistant named "CalmBot".
Provide thoughtful, calming, and helpful responses to users who are feeling stressed, anxious, or sad.
Never give medical advice. Just listen and offer support.

User: {user_input}
CalmBot:
"""

prompt = PromptTemplate(input_variables=["user_input"], template=template)

# Create the LangChain chain
chain = LLMChain(llm=llm, prompt=prompt)

# Interactive loop
print("🧘 CalmBot: Your friendly mental health support. Type 'exit' to stop.")
print("Note: This is not a substitute for professional medical advice.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("CalmBot: Take care! Remember to be kind to yourself.")
        break

    response = chain.invoke({"user_input": user_input})
    print("CalmBot:", response["text"])
    