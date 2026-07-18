from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(model="gemini-3.1-flash-lite")

topicPrompt = PromptTemplate(
    template="Generate Detaild Report About topic \n {topic}",
    input_variables=["topic"]
)

summaryPrompt = PromptTemplate(
    template="Generate brief Sumary about below report \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = topicPrompt | model | parser | summaryPrompt | model | parser

result = chain.invoke({"topic":"recently ind vs eng odi match"})

print(result)