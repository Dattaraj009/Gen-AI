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

# Apply the first prompt
prompt1 = templet1.format(topic="Artificial Intelligence")
result = model.invoke(prompt1)

# Apply the second prompt
prompt2 = templet2.format(text=result.content)
result1 = model.invoke(prompt2)

# Output
print(result1.content)

