"""
AI Doctor Crew - Medical Document Processing and Analysis System

This package provides a comprehensive CrewAI-based system for medical document 
processing, analysis, and information retrieval using ChromaDB vector database.

Main Components:
- DocCrew: Main crew class with medical agents and tasks
- ChromaDBClient: Vector database client for medical document storage
- Medical Agents: Specialized agents for medical query processing, document analysis, and reporting
- Medical Tasks: Configured tasks for medical workflows
"""

from .doc_crew import DocCrew
from .tools.milvus import (
    ChromaDBClient,
    search_chromadb_by_text,
    chromadb_search_tool,
    # Backward compatibility
    search_milvus_by_text,
    milvus_search_tool
)

__version__ = "0.1.0"
__author__ = "AI Doctor Crew Team"

__all__ = [
    "DocCrew",
    "ChromaDBClient", 
    "search_chromadb_by_text",
    "chromadb_search_tool",
    # Backward compatibility
    "search_milvus_by_text",
    "milvus_search_tool"
]
