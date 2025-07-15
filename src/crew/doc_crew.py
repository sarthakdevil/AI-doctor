#!/usr/bin/env python3
"""
AI Doctor Crew - Medical Document Processing and Analysis System
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process,LLM
from crewai.project import CrewBase, agent, crew, task

# Import tools
from tools.milvus import (
    ChromaDBClient,
    chromadb_search_tool,
    milvus_search_tool  # Backward compatibility alias
)
from tools.bytez import run_model
from tools.serper import SerperDevTool
load_dotenv()
os.getenv("GEMINI_API_KEY")
llm = LLM(
    model = "gemini/gemini-2.0-flash"
)

@CrewBase
class DocCrew:
    """AI Doctor Crew for medical document processing and analysis"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self):
        try:
            self.config_path = Path(__file__).parent / "config"
            self.agents_config_data = self._load_config("agents.yaml")
            self.tasks_config_data = self._load_config("tasks.yaml")
            
            # Initialize ChromaDB tools with safety checks
            self.milvus_tools = []
            if milvus_search_tool is not None:
                self.milvus_tools.append(milvus_search_tool)
                print("✅ ChromaDB search tool initialized successfully")
            else:
                print("⚠️ Warning: milvus_search_tool is None - ChromaDB may not be properly initialized")
                # Continue without the tool - crew can still function
            
            # Initialize other tools with error handling
            self.other_tools = []
            try:
                self.other_tools.append(SerperDevTool())
                print("✅ Serper search tool initialized")
            except Exception as serper_error:
                print(f"⚠️ Warning: Failed to initialize Serper tool: {serper_error}")
            
            try:
                self.other_tools.append(run_model)
                print("✅ Bytez model tool initialized")
            except Exception as bytez_error:
                print(f"⚠️ Warning: Failed to initialize Bytez tool: {bytez_error}")
                
            print(f"✅ DocCrew initialized with {len(self.milvus_tools)} ChromaDB tools and {len(self.other_tools)} other tools")
            
        except Exception as init_error:
            print(f"❌ Error during DocCrew initialization: {init_error}")
            print(f"Error type: {type(init_error).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = self.config_path / filename
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file) or {}
                print(f"Loaded {filename}: {list(config_data.keys())}")
                return config_data
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return {}
    
    @agent
    def medical_query_specialist(self) -> Agent:
        """Medical Query Specialist Agent"""
        return Agent(
            config=self.agents_config_data.get('medical_query_specialist', {}),
            tools=self.milvus_tools + self.other_tools,  # All tools for comprehensive analysis
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @agent
    def clinical_advisor(self) -> Agent:
        """Clinical Information Advisor Agent"""
        return Agent(
            config=self.agents_config_data.get('clinical_advisor', {}),
            tools=self.other_tools,  # Bytez medical LLM + Serper for internet research
            verbose=True,
            allow_delegation=True,
            max_iter=3,
            max_rpm=10,
            llm=llm
        )
    
    @agent
    def medical_researcher(self) -> Agent:
        """Medical Research Analyst Agent"""
        return Agent(
            config=self.agents_config_data.get('medical_researcher', {}),
            tools=self.milvus_tools,  # Vector DB search tools for patient documents
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @agent
    def output_synthesizer(self) -> Agent:
        """Medical Report Synthesizer Agent"""
        return Agent(
            config=self.agents_config_data.get('output_synthesizer', {}),
            verbose=True,  # No tools needed - just synthesizes information
            allow_delegation=False,
            max_iter=2,
            llm=llm
        )
    
    @task
    def analyze_medical_query_task(self) -> Task:
        """Analyze Medical Query Task"""
        return Task(
            config=self.tasks_config_data.get('analyze_medical_query', {}),
            agent=self.medical_query_specialist()
        )
    
    @task
    def search_patient_history_task(self) -> Task:
        """Search Patient History Task"""
        return Task(
            config=self.tasks_config_data.get('search_patient_history', {}),
            agent=self.medical_researcher(),
            context=[self.analyze_medical_query_task()]
        )
    
    @task
    def provide_medical_guidance_task(self) -> Task:
        """Provide Medical Guidance Task"""
        return Task(
            config=self.tasks_config_data.get('provide_medical_guidance', {}),
            agent=self.clinical_advisor(),
            context=[
                self.search_patient_history_task(),
                self.analyze_medical_query_task()
            ]
        )
    
    @task
    def synthesize_response_task(self) -> Task:
        """Synthesize Medical Response Task"""
        return Task(
            config=self.tasks_config_data.get('synthesize_response', {}),
            agent=self.output_synthesizer(),
            context=[
                self.provide_medical_guidance_task(),
                self.search_patient_history_task(),
                self.analyze_medical_query_task()
            ]
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the AI Doctor Crew"""
        return Crew(
            agents= self.agents,
            tasks= self.tasks,
            process=Process.sequential,
            verbose=True,
        )