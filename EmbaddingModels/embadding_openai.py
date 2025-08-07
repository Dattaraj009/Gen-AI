from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embadding=OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=32
)
result=embadding.embed_query("What is the capital of India")

print(str(result))