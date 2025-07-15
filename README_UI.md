# 🏥 AI Doctor PDF Vector Database UI

A modern, user-friendly web interface for uploading medical PDF documents and searching them using AI-powered semantic search.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-00ADD8?style=for-the-badge&logo=milvus&logoColor=white)

## ✨ Features

- 📤 **Drag & Drop PDF Upload**: Intuitive file upload interface
- 🤖 **AI-Powered Search**: Natural language semantic search using all-MiniLM-L6-v2
- 📊 **Real-time Processing**: Live progress tracking during PDF processing
- 🏷️ **Smart Metadata**: Automatic categorization and tagging
- 📈 **Analytics Dashboard**: Processing statistics and search history
- 🔍 **Advanced Search**: Configurable result count and filtering
- 💾 **Persistent Storage**: Documents stored in Milvus vector database
- 🎨 **Modern UI**: Clean, responsive design with dark/light mode

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Milvus database (local or cloud)
- Required environment variables set

### Installation

1. **Install Dependencies** (if not already done):
   ```bash
   pip install streamlit pymilvus sentence-transformers PyMuPDF numpy pandas
   ```

2. **Set Environment Variables**:
   Create a `.env` file with:
   ```bash
   MILVUS_ID=your_username
   MILVUS_PASSWORD=your_password
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   ```

3. **Launch the Application**:
   
   **Option A - Windows (Recommended):**
   ```bash
   # Double-click start_ui.bat
   # OR run in PowerShell:
   .\start_ui.ps1
   ```
   
   **Option B - Command Line:**
   ```bash
   python run_ui.py
   ```
   
   **Option C - Direct Streamlit:**
   ```bash
   streamlit run src/crew/main.py
   ```

4. **Open in Browser**:
   The app will automatically open at `http://localhost:8501`

## 🎯 How to Use

### 1. Upload PDF Documents

1. Navigate to the **"📤 Upload PDF"** tab
2. Configure processing settings in the sidebar:
   - **Chunk Size**: 200-1500 characters (recommended: 500)
   - **Overlap**: 0-300 characters (recommended: 50)
3. Set document metadata:
   - Category (medical_report, research_paper, etc.)
   - Department (cardiology, neurology, etc.)
   - Classification (public, confidential, restricted)
   - Custom tags
4. Drag and drop PDF files or click to browse
5. Click **"🚀 Process Files"** to upload to the vector database

### 2. Search Documents

1. Go to the **"🔍 Search Documents"** tab
2. Enter your search query in natural language:
   - "symptoms of heart disease"
   - "treatment for diabetes"
   - "chest pain diagnosis"
   - "blood pressure management"
3. Set the number of results (1-20)
4. Click **"🔍 Search"** to find relevant documents
5. View results with:
   - Relevance scores
   - Source document information
   - Page and chunk details
   - Full metadata

### 3. View Statistics

1. Visit the **"📊 Database Stats"** tab
2. Click **"📊 Refresh Stats"** to see:
   - Number of files processed
   - Search activity
   - Document chunk estimates
   - Processing history
   - System configuration

## 🎨 User Interface Overview

### Sidebar Features
- **🔧 Configuration**: Database connection status
- **⚙️ Processing Settings**: Chunk size and overlap controls
- **🏷️ Document Metadata**: Categorization options
- **📈 Processing History**: Recent file uploads

### Main Tabs

#### 📤 Upload PDF
- Multi-file upload support
- Real-time processing progress
- Success/error feedback
- File size and format validation

#### 🔍 Search Documents
- Natural language query input
- Configurable result count
- Detailed result display with expandable sections
- Search history tracking

#### 📊 Database Stats
- Processing metrics
- Search analytics
- System configuration info
- Processed files table

## 🔧 Configuration Options

### Processing Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Chunk Size | 200-1500 | 500 | Maximum characters per text chunk |
| Overlap | 0-300 | 50 | Overlapping characters between chunks |

### Document Categories

- **medical_report**: Patient reports, diagnoses
- **research_paper**: Academic publications
- **guidelines**: Treatment protocols
- **case_study**: Clinical case studies
- **other**: General medical documents

### Departments

- **cardiology**: Heart and cardiovascular
- **neurology**: Brain and nervous system
- **oncology**: Cancer treatment
- **radiology**: Medical imaging
- **general_medicine**: General practice
- **other**: Other specialties

## 🛠️ Troubleshooting

### Common Issues

#### "Failed to connect to Milvus"
- Check if Milvus server is running
- Verify MILVUS_ID and MILVUS_PASSWORD are correct
- Ensure MILVUS_HOST and MILVUS_PORT are accessible

#### "PDF processing failed"
- Ensure PDF file is not corrupted
- Check if PDF is password-protected
- Verify file size is reasonable (< 100MB recommended)

#### "Import errors"
- Install missing dependencies: `pip install -r requirements.txt`
- Check Python version compatibility (3.13+)

#### Application won't start
- Verify Streamlit is installed: `pip install streamlit`
- Check if port 8501 is available
- Try running with: `streamlit run src/crew/main.py --server.port 8502`

### Debug Mode

Enable detailed logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📱 Mobile Compatibility

The UI is responsive and works on:
- 💻 Desktop browsers
- 📱 Mobile devices
- 📋 Tablets

## 🔒 Security Features

- Environment variable protection
- Secure file handling with temporary files
- No data persistence in browser
- Metadata sanitization

## 🚀 Performance Tips

1. **Optimal Chunk Sizes**:
   - Medical reports: 500-800 characters
   - Research papers: 800-1200 characters
   - Guidelines: 400-600 characters

2. **Batch Processing**:
   - Upload multiple files at once
   - Use consistent metadata for related documents

3. **Search Optimization**:
   - Use specific medical terminology
   - Include context in queries
   - Adjust result count based on needs

## 🤝 Support

For issues or questions:
1. Check this README for common solutions
2. Review the Milvus connection status
3. Verify all dependencies are installed
4. Check environment variable configuration

## 📊 System Requirements

- **Memory**: 4GB+ RAM recommended
- **Storage**: Depends on document volume
- **Network**: Stable connection to Milvus server
- **Browser**: Modern browser with JavaScript enabled

---

*Built with ❤️ using Streamlit, Milvus, and all-MiniLM-L6-v2*
