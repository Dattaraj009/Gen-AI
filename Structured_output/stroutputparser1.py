from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Step 1: Prompt to generate a detailed report
templet1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"],
)

# Step 2: Prompt to generate summary
templet2 = PromptTemplate(
    template="Write a 5-line summary of the following report:\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = templet1 | model | parser | templet2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)

