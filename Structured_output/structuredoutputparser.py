from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",  
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


schema = [
    ResponseSchema(name='fact_1',description='fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)

chain = template | model | parser

result = chain.invoke({'topic':'Artificial Intelligence'})

print(result)


# prompt = templet.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# alternativs of above three lines are chain

