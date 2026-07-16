from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("write poem about friendship but only in 4 line")

print(response.content)