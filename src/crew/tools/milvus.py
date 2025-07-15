import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Try to import pypdf with better error handling
try:
    from pypdf import PdfReader
    PDF_PROCESSOR_AVAILABLE = True
    PDF_PROCESSOR_TYPE = "pypdf"
except ImportError as e:
    print(f"Warning: pypdf not available: {e}")
    print("Installing pypdf...")
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        from pypdf import PdfReader
        PDF_PROCESSOR_AVAILABLE = True
        PDF_PROCESSOR_TYPE = "pypdf"
        print("pypdf installed successfully!")
    except Exception as install_error:
        print(f"Failed to install pypdf: {install_error}")
        # Try alternative PDF processor
        try:
            import PyPDF2
            PDF_PROCESSOR_AVAILABLE = True
            PDF_PROCESSOR_TYPE = "PyPDF2"
            print("Using PyPDF2 as fallback PDF processor")
        except ImportError:
            PDF_PROCESSOR_AVAILABLE = False
            PDF_PROCESSOR_TYPE = None
            print("No PDF processor available. Please install pypdf or PyPDF2")

# Try to import ChromaDB with automatic installation
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ChromaDB not available: {e}")
    print("Installing ChromaDB...")
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb"])
        import chromadb
        from chromadb.config import Settings
        CHROMADB_AVAILABLE = True
        print("ChromaDB installed successfully!")
    except Exception as install_error:
        print(f"Failed to install ChromaDB: {install_error}")
        CHROMADB_AVAILABLE = False

from crewai.tools import BaseTool
from pydantic import Field
from dotenv import load_dotenv
import hashlib
import re
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBClient:
    """Client for interacting with ChromaDB vector database"""
    
    def __init__(self, collection_name: str = "medical_documents", persist_directory: str = None):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not available. Please install ChromaDB with: pip install chromadb\n"
                "This is required for document storage and search functionality."
            )
            
        self.collection_name = collection_name
        
        # Set default persist directory
        if persist_directory is None:
            persist_directory = os.path.join(os.getcwd(), "chroma_db")
        
        self.persist_directory = persist_directory
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self._connect()
        self._setup_collection()
    
    def _connect(self):
        """Connect to ChromaDB with persistent storage"""
        try:
            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Connected to ChromaDB at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def _setup_collection(self):
        """Set up the collection if it doesn't exist"""
        try:
            # Get list of existing collections
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if self.collection_name in existing_collections:
                # Collection exists, get it
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists with {self.collection.count()} documents")
            else:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Collection for storing medical document embeddings"}
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file page by page using pypdf
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            List of dictionaries with page content and metadata
        """
        if not PDF_PROCESSOR_AVAILABLE:
            raise ImportError(
                "No PDF processor is available. Please install pypdf with: pip install pypdf\n"
                "This is required for PDF text extraction functionality."
            )
            
        try:
            pages_content = []
            
            if PDF_PROCESSOR_TYPE == "pypdf":
                # Use pypdf (recommended)
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            text = page.extract_text()
                            
                            # Clean the text
                            text = re.sub(r'\s+', ' ', text.strip())
                            
                            if text:  # Only include pages with content
                                pages_content.append({
                                    'page_number': page_num + 1,
                                    'content': text,
                                    'char_count': len(text),
                                    'metadata': {
                                        'source_file': os.path.basename(pdf_path),
                                        'full_path': pdf_path,
                                        'page_count': total_pages,
                                        'pdf_processor': 'pypdf'
                                    }
                                })
                        except Exception as page_error:
                            logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                            continue
            
            elif PDF_PROCESSOR_TYPE == "PyPDF2":
                # Fallback to PyPDF2
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            text = page.extract_text()
                            
                            # Clean the text
                            text = re.sub(r'\s+', ' ', text.strip())
                            
                            if text:  # Only include pages with content
                                pages_content.append({
                                    'page_number': page_num + 1,
                                    'content': text,
                                    'char_count': len(text),
                                    'metadata': {
                                        'source_file': os.path.basename(pdf_path),
                                        'full_path': pdf_path,
                                        'page_count': total_pages,
                                        'pdf_processor': 'PyPDF2'
                                    }
                                })
                        except Exception as page_error:
                            logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                            continue
            
            logger.info(f"Extracted text from {len(pages_content)} pages from {pdf_path} using {PDF_PROCESSOR_TYPE}")
            return pages_content
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return []
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start + chunk_size // 2:
                        end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def _generate_document_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique document ID based on content and metadata"""
        identifier = f"{metadata.get('source_file', 'unknown')}_{content[:100]}"
        return hashlib.md5(identifier.encode()).hexdigest()[:16]
    
    def process_pdf_and_save(self, pdf_path: str, chunk_size: int = 500, 
                           overlap: int = 50, custom_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process a PDF file, extract text, and save to ChromaDB for text search
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Maximum characters per text chunk
            overlap: Overlapping characters between chunks
            custom_metadata: Additional metadata to attach to documents
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            # Extract text from PDF
            pages_content = self._extract_text_from_pdf(pdf_path)
            if not pages_content:
                logger.error("No content extracted from PDF")
                return False
            
            # Process each page and create chunks
            documents = []
            
            for page_data in pages_content:
                page_text = page_data['content']
                page_number = page_data['page_number']
                
                # Create text chunks
                chunks = self._chunk_text(page_text, chunk_size, overlap)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Create document metadata
                    doc_metadata = page_data['metadata'].copy()
                    if custom_metadata:
                        doc_metadata.update(custom_metadata)
                    
                    doc_metadata.update({
                        'chunk_size': len(chunk),
                        'chunk_index': chunk_idx,
                        'total_chunks_in_page': len(chunks),
                        'processing_date': datetime.now().isoformat(),
                        'page_number': page_number
                    })
                    
                    # Create document
                    document = {
                        'id': str(uuid.uuid4()),  # ChromaDB needs string IDs
                        'document_id': self._generate_document_id(chunk, doc_metadata),
                        'content': chunk,
                        'metadata': doc_metadata
                    }
                    
                    documents.append(document)
            
            # Save all documents to ChromaDB
            success = self.save_documents(documents)
            
            if success:
                logger.info(f"Successfully processed PDF {pdf_path} into {len(documents)} document chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return False
    
    
    def save_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Save documents to ChromaDB for text search
        
        Args:
            documents: List of dictionaries containing:
                - id: Unique identifier for the document
                - content: Text content
                - metadata: Additional metadata (optional)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to save")
                return False
            
            logger.info(f"Attempting to save {len(documents)} documents to ChromaDB")
            
            # Prepare data for ChromaDB
            ids = []
            documents_text = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                try:
                    if not all(key in doc for key in ['id', 'content']):
                        logger.error(f"Missing required fields in document {i}: {list(doc.keys())}")
                        continue
                    
                    ids.append(doc['id'])
                    documents_text.append(doc['content'])
                    metadatas.append(doc.get('metadata', {}))
                    
                except Exception as doc_error:
                    logger.error(f"Error processing document {i}: {doc_error}")
                    continue
            
            if not ids:
                logger.error("No valid documents to insert after validation")
                return False
            
            logger.info(f"Validated {len(ids)} documents for insertion")
            
            # Insert data to ChromaDB
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents_text,
                    metadatas=metadatas
                )
                
                logger.info(f"Successfully inserted {len(ids)} documents to ChromaDB")
                return True
                
            except Exception as insert_error:
                logger.error(f"Failed during insertion: {insert_error}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text query with ChromaDB
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return
        
        Returns:
            List of matching documents with similarity scores
        """
        try:
            if not query_text:
                logger.warning("Empty query text provided")
                return []
            
            # Perform text search using ChromaDB's built-in search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0.0] * len(results['documents'][0])
                )):
                    formatted_results.append({
                        "id": results['ids'][0][i] if results['ids'] else f"doc_{i}",
                        "content": doc,
                        "metadata": metadata,
                        "score": 1.0 - distance if distance else 1.0,  # Convert distance to similarity score
                        "distance": distance
                    })
            
            logger.info(f"Found {len(formatted_results)} documents for query: '{query_text[:50]}...'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search by text: {e}")
            return []
    
    def reset_collection(self):
        """Delete and recreate the collection (useful for clearing all data)"""
        try:
            # Delete collection if it exists
            existing_collections = [col.name for col in self.client.list_collections()]
            if self.collection_name in existing_collections:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection '{self.collection_name}'")
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Collection for storing medical document embeddings"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def get_collection_info(self):
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
        

class ChromaDBReaderTool(BaseTool):
    """Tool for semantic search in ChromaDB vector database"""
    
    name: str = "chromadb_search"
    description: str = "Search medical documents using text queries for semantic similarity matching"
    client: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = ChromaDBClient()
    
    def _run(self, query_text: str, top_k: int = 10) -> str:
        """
        Search for similar documents using text query
        
        Args:
            query_text: Text to search for in medical documents
            top_k: Number of top results to return (default: 10)
        
        Returns:
            String with search results
        """
        try:
            if not query_text:
                return "Error: query_text is required for search operation"
            
            results = self.client.search_by_text(
                query_text=query_text,
                top_k=top_k
            )
            
            if results:
                return f"Found {len(results)} documents matching text query '{query_text}': {results}"
            else:
                return f"No documents found for query: {query_text}"
                
        except Exception as e:
            return f"Error executing search operation: {str(e)}"





# Convenience functions for external use
def search_chromadb_by_text(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function to search by text in ChromaDB
    
    Args:
        query_text: Text query
        top_k: Number of results to return
    
    Returns:
        List of matching documents
    """
    try:
        client = ChromaDBClient()
        return client.search_by_text(query_text, top_k)
    except Exception as e:
        logger.error(f"Failed to search by text: {e}")
        return []


# Create tool instance for use in CrewAI
chromadb_search_tool = ChromaDBReaderTool()

# Backward compatibility aliases
search_milvus_by_text = search_chromadb_by_text
milvus_search_tool = chromadb_search_tool