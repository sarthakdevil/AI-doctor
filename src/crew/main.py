import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

# Import our ChromaDB tools
from tools.milvus import (
    ChromaDBClient,
    search_chromadb_by_text,
    chromadb_search_tool,
    # Backward compatibility aliases
    search_milvus_by_text,
    milvus_search_tool
)

# Page configuration
st.set_page_config(
    page_title="AI Doctor - PDF Vector Database",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

def check_milvus_connection():
    """Check if Milvus connection is properly configured"""
    required_vars = ["MILVUS_URI", "MILVUS_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please set MILVUS_URI and MILVUS_TOKEN in your environment variables.")
        return False
    
    try:
        # Test connection by creating a client
        client = ChromaDBClient()
        st.success("✅ Successfully connected to ChromaDB!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to connect to ChromaDB: {str(e)}")
        return False

def process_uploaded_pdf(uploaded_file, chunk_size: int, overlap: int, custom_metadata: Dict[str, Any]):
    """Process uploaded PDF file and save to vector database"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Add processing timestamp to metadata
        custom_metadata.update({
            "upload_timestamp": datetime.now().isoformat(),
            "file_size_bytes": uploaded_file.size,
            "original_filename": uploaded_file.name
        })
        
        # Process PDF
        success = process_pdf_to_milvus(
            pdf_path=tmp_file_path,
            chunk_size=chunk_size,
            overlap=overlap,
            custom_metadata=custom_metadata
        )
        
        if success:
            # Add to processed files list
            st.session_state.processed_files.append({
                "filename": uploaded_file.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "metadata": custom_metadata
            })
        
        return success
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def display_search_results(results: List[Dict[str, Any]]):
    """Display search results in a formatted way"""
    
    if not results:
        st.info("No results found.")
        return
    
    st.write(f"Found **{len(results)}** relevant documents:")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"Result {i} - Score: {result['score']:.3f}", expanded=i <= 3):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Content:**")
                st.write(result['content'])
                
            with col2:
                st.markdown("**Document Info:**")
                metadata = result.get('metadata', {})
                
                info_data = {
                    "Source File": metadata.get('source_file', 'Unknown'),
                    "Page": result.get('page_number', 'N/A'),
                    "Chunk": result.get('chunk_index', 'N/A'),
                    "Category": metadata.get('category', 'N/A'),
                    "Department": metadata.get('department', 'N/A'),
                    "Document Type": metadata.get('document_type', 'N/A')
                }
                
                for key, value in info_data.items():
                    st.write(f"**{key}:** {value}")
                
                # Show additional metadata if available
                if metadata:
                    with st.expander("Full Metadata"):
                        st.json(metadata)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏥 AI Doctor - PDF Vector Database")
    st.markdown("Upload medical documents and search them using AI-powered semantic search")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Milvus connection status
        st.subheader("Database Connection")
        if st.button("Check Connection"):
            check_milvus_connection()
        
        st.divider()
        
        # Processing settings
        st.subheader("Processing Settings")
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=500, step=50,
                              help="Maximum characters per text chunk")
        overlap = st.slider("Overlap", min_value=0, max_value=300, value=50, step=10,
                           help="Overlapping characters between chunks")
        
        st.divider()
        
        # Document metadata
        st.subheader("Document Metadata")
        category = st.selectbox("Category", ["medical_report", "research_paper", "guidelines", "case_study", "other"])
        department = st.selectbox("Department", ["cardiology", "neurology", "oncology", "radiology", "general_medicine", "other"])
        classification = st.selectbox("Classification", ["public", "confidential", "restricted"])
        
        # Custom tags
        custom_tags = st.text_input("Custom Tags (comma-separated)", placeholder="heart, diagnosis, treatment")
        
        # Processing history
        st.divider()
        st.subheader("📈 Processing History")
        if st.session_state.processed_files:
            st.write(f"Processed: {len(st.session_state.processed_files)} files")
            for file_info in st.session_state.processed_files[-3:]:  # Show last 3
                st.write(f"• {file_info['filename']} ({file_info['timestamp']})")
        else:
            st.write("No files processed yet")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "🔍 Search Documents", "📊 Database Stats"])
    
    with tab1:
        st.header("Upload PDF Documents")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF files to upload to the vector database"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            # Display file info
            for uploaded_file in uploaded_files:
                st.write(f"📄 **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            # Process button
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🚀 Process Files", type="primary"):
                    # Prepare metadata
                    custom_metadata = {
                        "category": category,
                        "department": department,
                        "classification": classification,
                        "processed_by": "streamlit_ui",
                        "tags": [tag.strip() for tag in custom_tags.split(",") if tag.strip()]
                    }
                    
                    # Check connection first
                    if not check_milvus_connection():
                        st.stop()
                    
                    # Process each file
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    success_count = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        try:
                            success = process_uploaded_pdf(uploaded_file, chunk_size, overlap, custom_metadata)
                            
                            if success:
                                success_count += 1
                                st.success(f"✅ Successfully processed {uploaded_file.name}")
                            else:
                                st.error(f"❌ Failed to process {uploaded_file.name}")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    status_text.text("Processing complete!")
                    st.info(f"Successfully processed {success_count} out of {len(uploaded_files)} files")
            
            with col2:
                if st.button("🗑️ Clear Selection"):
                    st.rerun()
    
    with tab2:
        st.header("Search Documents")
        
        # Search interface
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'symptoms of heart disease', 'treatment for diabetes', 'chest pain diagnosis'",
            help="Use natural language to search through your documents"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        with col2:
            if st.button("🔍 Search", type="primary", disabled=not search_query):
                if not check_milvus_connection():
                    st.stop()
                
                with st.spinner("Searching documents..."):
                    try:
                        results = search_milvus_by_text(search_query, top_k=search_top_k)
                        
                        # Add to search history
                        st.session_state.search_history.append({
                            "query": search_query,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "results_count": len(results)
                        })
                        
                        # Display results
                        display_search_results(results)
                        
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
        
        # Search history
        if st.session_state.search_history:
            st.divider()
            st.subheader("Recent Searches")
            
            # Show last 5 searches
            recent_searches = st.session_state.search_history[-5:]
            for search in reversed(recent_searches):
                with st.expander(f"'{search['query'][:50]}...' - {search['results_count']} results"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Query:** {search['query']}")
                        st.write(f"**Results:** {search['results_count']} documents found")
                    with col2:
                        st.write(f"**Time:** {search['timestamp']}")
                        if st.button("🔄 Search Again", key=f"repeat_{search['timestamp']}"):
                            st.text_input("Enter your search query:", value=search['query'], key="repeated_search")
    
    with tab3:
        st.header("Database Statistics")
        
        if st.button("📊 Refresh Stats"):
            if not check_milvus_connection():
                st.stop()
            
            try:
                # Use MCP tool to get statistics
                result = milvus_reader_tool._run(operation="get_by_id", document_id="dummy")  # This will show connection works
                
                # Display processing statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Files Processed", len(st.session_state.processed_files))
                
                with col2:
                    st.metric("Searches Performed", len(st.session_state.search_history))
                
                with col3:
                    # Calculate total chunks (approximate)
                    total_estimated_chunks = sum([
                        max(1, file_info.get('file_size', 1000) // file_info.get('chunk_size', 500))
                        for file_info in st.session_state.processed_files
                    ])
                    st.metric("Estimated Document Chunks", total_estimated_chunks)
                
                # Processed files table
                if st.session_state.processed_files:
                    st.subheader("Processed Files")
                    df = pd.DataFrame(st.session_state.processed_files)
                    st.dataframe(df, use_container_width=True)
                
                # Search history
                if st.session_state.search_history:
                    st.subheader("Search Analytics")
                    search_df = pd.DataFrame(st.session_state.search_history)
                    
                    # Most common search terms
                    all_queries = ' '.join(search_df['query'].tolist()).lower()
                    st.write("Recent search queries demonstrate the types of medical information being accessed.")
                
            except Exception as e:
                st.error(f"Failed to get statistics: {str(e)}")
        
        # Configuration info
        st.divider()
        st.subheader("System Configuration")
        
        config_info = {
            "Embedding Model": "all-MiniLM-L6-v2",
            "Vector Dimension": "384",
            "Database": "Milvus",
            "Chunking Strategy": "Sentence-aware with overlap",
            "Search Method": "Cosine similarity"
        }
        
        for key, value in config_info.items():
            st.write(f"**{key}:** {value}")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🏥 AI Doctor PDF Vector Database - Powered by Milvus & all-MiniLM-L6-v2
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()