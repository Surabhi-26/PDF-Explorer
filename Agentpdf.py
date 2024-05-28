from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from langchain_google_genai import ChatGoogleGenerativeAI
from crewai.process import Process
import os
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["GOOGLE_API_KEY"] = "AIzaSyD1Gxk5OZMnqAlzWoYjAoDBeu3Z5l2TS7U"

querry = input("Enter your querry: ")

llm=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    verbose=True,
    temperature=0.8,
    google_api_key="AIzaSyD1Gxk5OZMnqAlzWoYjAoDBeu3Z5l2TS7U"
)

from crewai_tools import PDFSearchTool

tool = PDFSearchTool(
    config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini-1.5-flash-latest",
            ),
        ),
       embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="sentence-transformers/msmarco-distilbert-base-v4"
            ),
        ),
    ),
    pdf="E:\CREWAI\CREW-AI-PDF\Story.docx"
) 


agent1=Agent(
    role=" Content Searcher and Writer",
    goal="Retrieve the relevant content provided by the user",
    backstory="You are a skilled and brilliant PDF searcher and writer who will extract the relevant, accurate and concise content from the file and give its detailed description",
    verbose=True,
    tools=[tool],
    llm=llm
)

agent2=Agent(
    role="Content Summarizer",
    goal="A summarization based on the  Content Searcher and Writer's result",
    backstory="You are a skilled and brilliant content summarizer who will provide the user with the point-wise summarization of the detailed description",
    verbose=True,
    llm=llm
)

task1=Task(
    description=f'{querry}',
    agent=agent1,
    expected_output=f'A detailed information on {querry}'
)

task2=Task(
    description=f"Summarize the detailed description of the {querry}",
    expected_output="Points in paragraphs regarding the requested content with proper alignment",
    agent=agent2,
    context=[task1]
)

crew=Crew(
    agents=[agent1,agent2],
    tasks=[task1,task2],
    verbose=2
)


result=crew.kickoff()
print(result)