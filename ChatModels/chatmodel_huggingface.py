from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Get your API token from the environment
# api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Create HuggingFace endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation",
    # huggingfacehub_api_token=api_key  # ✅ MUST include this
)
# print("Using model:", llm.repo_id)
# print("Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


# Wrap with chat interface
model = ChatHuggingFace(llm=llm)

# Send prompt
result = model.invoke("What is the capital of India")

# Print the content
print(result.content)
