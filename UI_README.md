# 🏥 AI Doctor - PDF Vector Database UI

A comprehensive Streamlit web interface for uploading medical PDF documents and performing semantic search using Milvus vector database with all-MiniLM-L6-v2 embeddings.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Milvus database** running and accessible
3. **Environment variables** configured

### Setup Environment Variables

Create a `.env` file in the project root:

```bash
MILVUS_URI=your_milvus_connection_uri
MILVUS_TOKEN=your_milvus_access_token
```

### Launch the Application

#### Option 1: Using the Launch Script (Recommended)
```bash
python launch_ui.py
```

#### Option 2: Using Batch File (Windows)
Double-click `launch_ui.bat` or run:
```cmd
launch_ui.bat
```

#### Option 3: Direct Streamlit Command
```bash
streamlit run main_ui.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🎯 Features

### 📤 PDF Upload & Processing
- **Multi-file Upload**: Upload multiple PDF files simultaneously
- **Intelligent Chunking**: Configurable text chunking with sentence boundary awareness
- **Rich Metadata**: Add categories, departments, classifications, and custom tags
- **Progress Tracking**: Real-time processing progress with status updates
- **Error Handling**: Comprehensive error reporting and recovery

### 🔍 Semantic Search
- **Natural Language Queries**: Search using plain English questions
- **Similarity Scoring**: Results ranked by semantic similarity
- **Category Filtering**: Filter results by document categories
- **Search History**: Track and repeat previous searches
- **Rich Results Display**: Detailed result cards with metadata

### 📊 Database Analytics
- **Collection Statistics**: View total documents and embedding dimensions
- **Processing History**: Charts showing upload patterns over time
- **Category Distribution**: Visual breakdown of document categories
- **Search Analytics**: Track search patterns and performance
- **System Information**: Environment and connection status

## 🖥️ User Interface Guide

### Sidebar Configuration

#### Database Connection
- **Check Connection**: Test Milvus connectivity
- **Environment Status**: View URI and token configuration
- **Connection Indicators**: Visual status of database connection

#### Processing Settings
- **Chunk Size**: 200-1500 characters (default: 500)
- **Overlap**: 0-300 characters (default: 50)
- **Smart Chunking**: Automatic sentence boundary detection

#### Document Metadata
- **Category**: medical_report, research_paper, guidelines, case_study, other
- **Department**: cardiology, neurology, oncology, radiology, general_medicine, other  
- **Classification**: public, confidential, restricted
- **Custom Tags**: Comma-separated keywords

#### Processing History
- **Recent Files**: Last 3 processed files
- **File Count**: Total processed files counter
- **Clear History**: Reset processing history

### Main Tabs

#### 📤 Upload PDF Tab
1. **File Selection**: Choose one or more PDF files
2. **File Preview**: View selected files with size information
3. **Process Files**: Start PDF processing with current settings
4. **Progress Tracking**: Real-time progress bar and status updates
5. **Results Summary**: Success/failure status for each file

#### 🔍 Search Documents Tab
1. **Query Input**: Enter natural language search queries
2. **Result Count**: Configure number of results (1-20)
3. **Category Filter**: Optional filtering by document category
4. **Search Execution**: Perform semantic search
5. **Results Display**: Expandable result cards with:
   - Content preview
   - Similarity score
   - Source file information
   - Page and chunk details
   - Full metadata view
6. **Search History**: Recent searches with repeat functionality

#### 📊 Database Stats Tab
1. **Key Metrics**: 
   - Total documents in database
   - Embedding dimensions
   - Collection name
   - Model information
2. **Processing Charts**:
   - Files processed over time
   - Category distribution
3. **Search Analytics**:
   - Total searches performed
   - Recent search history table
4. **System Information**:
   - Environment variable status
   - Session state details
   - Version information

## 🔧 Configuration Options

### Chunk Size Guidelines
- **Medical Reports**: 500-800 characters
- **Research Papers**: 800-1200 characters
- **Guidelines**: 400-600 characters
- **Case Studies**: 600-1000 characters

### Overlap Settings
- **Technical Documents**: 100-200 characters
- **General Medical Text**: 50-100 characters
- **Narrative Content**: 50-80 characters

### Search Parameters
- **Top-K Results**: 1-20 (recommended: 5-10)
- **Category Filtering**: Use for focused searches
- **Query Length**: Optimal 5-50 words

## 📝 Usage Examples

### Example Search Queries
- "symptoms of heart disease in elderly patients"
- "side effects of ACE inhibitors"
- "diagnostic criteria for diabetes mellitus"
- "emergency treatment for myocardial infarction"
- "hypertension management guidelines"

### Metadata Best Practices
```json
{
    "category": "medical_report",
    "department": "cardiology",
    "classification": "confidential",
    "tags": ["heart", "diagnosis", "echocardiogram"],
    "patient_id": "anonymized",
    "report_date": "2025-01-15"
}
```

## 🔍 Troubleshooting

### Common Issues

#### Connection Problems
- **Check Environment Variables**: Ensure MILVUS_URI and MILVUS_TOKEN are set
- **Network Access**: Verify Milvus server is accessible
- **Authentication**: Confirm token has proper permissions

#### Upload Failures
- **File Format**: Only PDF files are supported
- **File Size**: Large files may take longer to process
- **File Corruption**: Ensure PDFs are not corrupted or password-protected

#### Search Issues
- **No Results**: Try broader queries or check if documents are uploaded
- **Slow Performance**: Reduce top-k value or optimize chunk size
- **Poor Relevance**: Adjust query phrasing or use category filters

### Error Messages

#### "Missing environment variables"
- Set MILVUS_URI and MILVUS_TOKEN in your environment
- Check .env file location and syntax

#### "Failed to connect to Milvus"
- Verify Milvus server is running
- Check network connectivity
- Validate URI and token format

#### "Failed to process PDF"
- Ensure PDF is not password-protected
- Check file format and integrity
- Verify sufficient disk space

### Performance Optimization

#### For Large Document Collections
- Use category filtering to narrow searches
- Optimize chunk size based on document type
- Consider batch processing for multiple files

#### For Better Search Results
- Use specific medical terminology
- Include context in queries
- Leverage metadata filtering

## 🛠️ Development

### Adding Custom Features

#### New Document Types
1. Add to category selectbox in sidebar
2. Update metadata schema if needed
3. Adjust chunk size recommendations

#### Enhanced Search Filters
1. Modify search interface in tab2
2. Update search logic with new filters
3. Add to results display

#### Additional Analytics
1. Create new metrics in database stats
2. Add visualization components
3. Update refresh functionality

### API Integration
The UI uses the following core functions:
- `MilvusClient()`: Database connection and operations
- `process_pdf_to_milvus()`: PDF processing pipeline
- `search_milvus_by_text()`: Semantic search functionality

## 📦 Dependencies

### Core Requirements
- `streamlit`: Web interface framework
- `pymilvus`: Milvus database client
- `sentence-transformers`: Embedding model
- `PyMuPDF`: PDF text extraction
- `pandas`: Data manipulation
- `numpy`: Numerical operations

### Optional Enhancements
- `plotly`: Interactive charts
- `streamlit-aggrid`: Enhanced data tables
- `streamlit-tags`: Better tag input

## 🔒 Security Considerations

### Data Privacy
- All document processing is local
- Embeddings remain in your Milvus instance
- No data sent to external services

### Access Control
- Use appropriate classification levels
- Implement user authentication if needed
- Monitor search queries for compliance

### Environment Security
- Keep tokens secure and rotated
- Use environment variables, not hardcoded values
- Implement proper network security

## 📈 Monitoring & Maintenance

### Regular Tasks
- Monitor database storage usage
- Review search analytics for insights
- Update embedding models as needed
- Clean up old or unused documents

### Performance Monitoring
- Track search response times
- Monitor upload success rates
- Analyze user search patterns
- Review error logs regularly

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Milvus documentation
3. Verify environment configuration
4. Check application logs in terminal

---

**Note**: This UI is designed for medical document processing. Ensure compliance with relevant healthcare data regulations (HIPAA, GDPR, etc.) when handling sensitive medical information.
