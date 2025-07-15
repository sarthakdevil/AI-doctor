# Milvus PDF Processing Quick Start Guide

This guide demonstrates how to use the enhanced Milvus tools for processing PDF documents and creating embeddings using the all-MiniLM-L6-v2 model.

## Features

✅ **PDF Text Extraction**: Automatically extract text from PDF files page by page  
✅ **Smart Text Chunking**: Split text into overlapping chunks with sentence boundary awareness  
✅ **Automatic Embeddings**: Generate embeddings using all-MiniLM-L6-v2 model  
✅ **Semantic Search**: Search documents using natural language queries  
✅ **Metadata Support**: Store and query document metadata  
✅ **MCP Tool Integration**: Use as CrewAI tool for AI agents  

## Quick Setup

1. **Install Dependencies**:
   ```bash
   pip install pymilvus sentence-transformers PyMuPDF numpy
   ```

2. **Set Environment Variables**:
   ```bash
   # In your .env file
   MILVUS_URI=your_milvus_uri
   MILVUS_TOKEN=your_milvus_token
   ```

## Basic Usage

### 1. Process a PDF File

```python
from tools.milvus import process_pdf_to_milvus

# Process PDF and save to Milvus
success = process_pdf_to_milvus(
    pdf_path="medical_report.pdf",
    chunk_size=500,
    overlap=50,
    custom_metadata={
        "category": "medical",
        "department": "cardiology"
    }
)
```

### 2. Search by Text Query

```python
from tools.milvus import search_milvus_by_text

# Search using natural language
results = search_milvus_by_text(
    query_text="symptoms of heart disease",
    top_k=5
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Page: {result['page_number']}")
```

### 3. Use as MCP Tool

```python
from tools.milvus import milvus_reader_tool

# Process PDF
result = milvus_reader_tool._run(
    operation="process_pdf",
    pdf_path="document.pdf",
    chunk_size=600
)

# Search by text
result = milvus_reader_tool._run(
    operation="search_text",
    query_text="treatment guidelines",
    top_k=3
)
```

### 4. Advanced Client Usage

```python
from tools.milvus import MilvusClient

# Initialize with custom embedding model
client = MilvusClient(embedding_model="all-MiniLM-L6-v2")

# Process PDF with custom settings
client.process_pdf_and_save(
    pdf_path="research_paper.pdf",
    chunk_size=1000,
    overlap=200,
    custom_metadata={
        "document_type": "research",
        "authors": ["Dr. Smith"],
        "year": "2025"
    }
)

# Search with filters
results = client.search_by_text(
    query_text="clinical trial results",
    top_k=10,
    document_id_filter="specific_doc_id"  # Optional filter
)
```

## Data Schema

The Milvus collection stores documents with the following fields:

- **id**: Auto-generated primary key
- **document_id**: Unique identifier (MD5 hash of content + metadata)
- **content**: Text chunk content
- **embedding**: 384-dimensional vector (all-MiniLM-L6-v2)
- **metadata**: JSON metadata object
- **page_number**: Source page number
- **chunk_index**: Chunk index within the page

## Search Operations

### Available Operations

1. **process_pdf**: Process PDF and save embeddings
2. **search_text**: Search using natural language query
3. **search**: Search using pre-computed embedding vector
4. **get_by_id**: Retrieve documents by document ID

### MCP Tool Parameters

```python
# Process PDF
milvus_reader_tool._run(
    operation="process_pdf",
    pdf_path="path/to/file.pdf",
    chunk_size=500,          # Optional, default: 500
    overlap=50,              # Optional, default: 50
    custom_metadata={}       # Optional
)

# Search by text
milvus_reader_tool._run(
    operation="search_text",
    query_text="search query",
    top_k=10,               # Optional, default: 10
    document_id_filter=None # Optional
)
```

## Best Practices

### Chunk Size Guidelines
- **Medical documents**: 500-800 characters
- **Research papers**: 800-1200 characters  
- **General documents**: 400-600 characters

### Overlap Settings
- **Technical documents**: 100-200 characters
- **General text**: 50-100 characters
- **Narrative text**: 50-80 characters

### Metadata Usage
```python
metadata = {
    "category": "medical_report",
    "department": "cardiology", 
    "classification": "confidential",
    "date": "2025-07-14",
    "keywords": ["heart", "diagnosis"],
    "author": "Dr. Smith",
    "version": "1.0"
}
```

## Error Handling

The tools include comprehensive error handling:

- **Connection errors**: Automatic retry with meaningful messages
- **PDF parsing errors**: Graceful handling of corrupted files
- **Embedding errors**: Fallback to alternative processing
- **Schema validation**: Automatic field validation

## Performance Tips

1. **Batch Processing**: Process multiple PDFs in batches
2. **Index Optimization**: Use appropriate Milvus index parameters
3. **Memory Management**: Process large PDFs in chunks
4. **Connection Pooling**: Reuse client connections

## Integration with CrewAI

```python
from crewai import Agent, Task, Crew
from tools.milvus import milvus_reader_tool

# Create agent with Milvus tool
medical_agent = Agent(
    role="Medical Document Analyst",
    goal="Analyze medical documents and provide insights",
    tools=[milvus_reader_tool],
    backstory="Expert in medical document analysis"
)

# Create task
analysis_task = Task(
    description="Search for information about heart disease symptoms",
    agent=medical_agent,
    expected_output="Detailed analysis with source references"
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install required dependencies
2. **Connection Failed**: Check Milvus server status and credentials
3. **PDF Processing Failed**: Ensure PDF is not corrupted or password-protected
4. **Memory Issues**: Reduce chunk size or process files individually

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check environment variables:
```python
import os
print(f"MILVUS_URI: {os.getenv('MILVUS_URI')}")
print(f"MILVUS_TOKEN: {os.getenv('MILVUS_TOKEN')}")
```
