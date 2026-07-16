from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
result = embeddings.embed_query("what is the weather like today?",output_dimensionality=768)


documentEmbedding = embeddings.embed_documents([ "Today is Monday",
        "Today is Tuesday",
        "Today is April Fools day",],output_dimensionality=768)


# print(result)
print(documentEmbedding)