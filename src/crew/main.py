from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import json
from doc_crew import DocCrew
from tools.milvus import ChromaDBClient, create_chromadb_client_safe

# Initialize FastAPI app
app = FastAPI()

# Initialize CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for structured input
class MedicalQuery(BaseModel):
    query: str
    age: Optional[str] = None
    gender: Optional[str] = None
    urgency: Optional[str] = "Medium"
    specialty: Optional[str] = "General"

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Doctor Crew API!"}

@app.post("/upload_and_embed")
async def upload_and_embed(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    overlap: int = 50,
    custom_metadata: Optional[str] = None
):
    """
    Upload a PDF file and create vector embeddings
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Write uploaded content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Parse custom metadata if provided
            metadata = {}
            if custom_metadata:
                try:
                    metadata = json.loads(custom_metadata)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON format for custom_metadata")
            
            # Add upload info to metadata
            metadata.update({
                "original_filename": file.filename,
                "upload_size": len(content),
                "content_type": file.content_type
            })
            
            # Create ChromaDB client
            client = create_chromadb_client_safe(reset_on_init=False)
            if client is None:
                raise HTTPException(status_code=500, detail="Failed to initialize ChromaDB client")
            
            # Process PDF and create embeddings
            success = client.process_pdf_and_save(
                pdf_path=temp_file_path,
                chunk_size=chunk_size,
                overlap=overlap,
                custom_metadata=metadata
            )
            
            if success:
                # Get collection info
                collection_info = client.get_collection_info()
                return {
                    "status": "success",
                    "message": f"Successfully processed uploaded PDF: {file.filename}",
                    "collection_info": collection_info,
                    "processed_chunks": collection_info.get("document_count", "unknown")
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to process uploaded PDF and create embeddings")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup temporary file: {cleanup_error}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during upload: {str(e)}")

@app.post("/run")
async def run_doc_crew(input: MedicalQuery):
    try:
        doc_crew = DocCrew()
        
        # Format the query with all available context
        formatted_query = f"Medical Query: {input.query}\n"
        
        if input.age:
            formatted_query += f"Patient Age: {input.age}\n"
        
        if input.gender:
            formatted_query += f"Patient Gender: {input.gender}\n"
        
        formatted_query += f"Urgency: {input.urgency}\n"
        formatted_query += f"Specialty: {input.specialty}"
        
        # Pass the formatted query to the crew
        result = doc_crew.crew().kickoff(inputs={"user_query": formatted_query})
        
        return {"result": result}
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)