import os
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()
# Initialize Serper tool
os.getenv("SERPER_API_KEY")  # Set your Serper API key
serper_tool = SerperDevTool()
