from dotenv import load_dotenv
import os 
from langchain_openai import AzureChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import agentops

load_dotenv()

OPENAI_API_GPT_4_KEY = os.getenv('OPENAI_API_GPT_4_KEY')
OPENAI_API_GPT_4_TYPE = os.getenv('OPENAI_API_GPT_4_TYPE')
OPENAI_API_GPT_4_BASE = os.getenv('OPENAI_API_GPT_4_BASE')
OPENAI_API_GPT_4_VERSION = os.getenv('OPENAI_API_GPT_4_VERSION')
DEPLOYMENT_NAME_GPT_4o = os.getenv('DEPLOYMENT_NAME_GPT_4o')
os.environ["AGENTOPS_API_KEY"] = os.getenv("AGENTOPS_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

agentops.init(tags=["crewai-agent"])

llm = AzureChatOpenAI(openai_api_version=OPENAI_API_GPT_4_VERSION,
    azure_deployment=DEPLOYMENT_NAME_GPT_4o,
    model="gpt-4o",
    temperature=0.1,
    openai_api_key=OPENAI_API_GPT_4_KEY,
    azure_endpoint=OPENAI_API_GPT_4_BASE
)

search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  llm=llm,
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  llm=llm
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2,
)

result = crew.kickoff()
print("The outputs have been compiled")
print("Result=> ", result)

agentops.end_session("Success")




