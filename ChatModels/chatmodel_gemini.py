from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # ✅ Correct name as per API (use 'gemini-1.5-pro' for now)
    temperature=0,
    max_output_tokens=10     # ✅ Correct param name is 'max_output_tokens'
)

response = model.invoke("What is the capital of India?")
print(response.content)
# ✅ Ensure you have the correct API key set in your environment variables