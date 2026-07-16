from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

response = model.invoke("Why do parrots talk?")

print(response.content)