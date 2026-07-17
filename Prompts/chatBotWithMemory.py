"""
Create basic chatbot which can store prev chat memory in file so everytime when you talk with llm it append data in text file so you also know about message placeholder and messages for  conversation
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

# create file if not exist
if not os.path.exists("chatHistory.txt"):
    with open("chatHistory.txt", "w"):
        pass


# create prompt template
chatTemplate = ChatPromptTemplate(
    [
        ("system", "You are helpfull and kind assistant which help on user query"),
        MessagesPlaceholder(variable_name="chatHistory"),
        ("human", "{userQuery}"),
    ]
)


ongoingChatHistory = []


# read file
with open("chatHistory.txt", "r") as file:
    ongoingChatHistory = file.readlines()

while True:
    userQuery = input("You: ")

    if userQuery == "exit":
        break

    ongoingChatHistory.append(HumanMessage(content=userQuery))

    with open("chatHistory.txt", "a") as file:
        file.write(f"Human: {userQuery}\n")

        prompt = chatTemplate.invoke(
            {"chatHistory": ongoingChatHistory, "userQuery": userQuery}
        )

        response = model.invoke(prompt)
        print(f"AI: {response.content[0]['text']}")
        ongoingChatHistory.append(AIMessage(content=response.content[0]["text"]))

        with open("chatHistory.txt", "a") as file:
            file.write(f"AI: {response.content[0]['text']}\n")


print("See you Again :)")
