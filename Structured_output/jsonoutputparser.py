from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",  
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

templet = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


# prompt = templet.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# alternativs of above three lines are chain

chain = templet | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
