# AI Doctor Crew 🏥

A comprehensive CrewAI-based system for medical document processing, analysis, and information retrieval using Milvus vector database and Google's Gemini embeddings.

## 🚀 Features

- **Medical Document Processing**: Upload and process PDF medical documents
- **Semantic Search**: Find relevant medical information using vector embeddings
- **Specialized Medical Agents**: Expert agents for different medical tasks
- **Clinical Decision Support**: AI-powered medical analysis and insights
- **Web Interface**: User-friendly Streamlit interface
- **Vector Database**: Milvus integration for efficient document storage and retrieval
- **Google Gemini Embeddings**: State-of-the-art embedding model for medical content

## 🏗️ Architecture

### Agents
- **Medical Query Specialist**: Creates precise medical queries for information retrieval
- **Document Processor**: Processes medical PDFs and creates searchable embeddings
- **Clinical Advisor**: Provides comprehensive medical insights and decision support
- **Medical Researcher**: Conducts thorough medical research using available documents
- **Output Synthesizer**: Generates structured medical reports and summaries

### Tasks
- **Medical Document Processing**: Extract and embed medical content
- **Query Analysis**: Understand and refine medical queries
- **Information Search**: Retrieve relevant medical information
- **Clinical Insights**: Provide medical analysis and recommendations
- **Report Generation**: Create comprehensive medical reports

## 📋 Prerequisites

- Python 3.11 or higher
- Google API key for Gemini embeddings
- Milvus server (local or cloud)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI-doctor/crew
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # Google API Configuration
   GOOGLE_API_KEY=your_google_api_key_here
   
   # Milvus Configuration
   MILVUS_URI=your_milvus_uri
   MILVUS_TOKEN=your_milvus_token
   ```

4. **Get API Keys**:
   - **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Milvus**: Set up [Milvus Cloud](https://cloud.zilliz.com/) or run locally

## 🚀 Quick Start

### 1. Test the Setup
```bash
python launch_crew.py
```

### 2. Run the Web Interface
```bash
streamlit run src/crew/main.py
```

### 3. Use in Your Code
```python
from crew import DocCrew

# Initialize the crew
doc_crew = DocCrew()

# Analyze a medical query
result = doc_crew.run_medical_analysis(
    "What are the symptoms and treatment options for hypertension?"
)
print(result)

# Process a medical document
doc_crew.process_medical_document("path/to/medical_document.pdf")

# Run symptom analysis
symptoms = "Patient has chest pain, shortness of breath, and fatigue"
result = doc_crew.run_symptom_analysis(symptoms)
print(result)
```

## 📊 Usage Examples

### Medical Query Analysis
```python
query = """
A 45-year-old patient presents with chest pain, shortness of breath, 
and fatigue. What are the potential differential diagnoses?
"""
result = doc_crew.run_medical_analysis(query)
```

### Drug Interaction Check
```python
medications = """
- Warfarin 5mg daily
- Aspirin 81mg daily  
- Metoprolol 50mg twice daily
"""
result = doc_crew.run_drug_interaction_check(medications)
```

### Document Processing
```python
# Process a medical PDF
result = doc_crew.process_medical_document("medical_guidelines.pdf")
```

## 🔧 Configuration

### Agents Configuration (`config/agents.yaml`)
Customize agent roles, goals, and backstories for your specific medical use case.

### Tasks Configuration (`config/tasks.yaml`)
Define custom medical workflows and task dependencies.

### Environment Variables
- `GOOGLE_API_KEY`: Your Google API key for Gemini embeddings
- `MILVUS_URI`: Milvus server URI
- `MILVUS_TOKEN`: Milvus authentication token

## 🏥 Medical Use Cases

1. **Clinical Decision Support**: Analyze symptoms and provide differential diagnoses
2. **Medical Literature Review**: Search and analyze medical research papers
3. **Drug Safety**: Check for drug interactions and contraindications
4. **Treatment Protocols**: Find evidence-based treatment guidelines
5. **Medical Education**: Educational content analysis and Q&A

## ⚠️ Medical Disclaimer

This system is designed for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📁 Project Structure

```
crew/
├── src/crew/
│   ├── __init__.py
│   ├── doc_crew.py          # Main crew implementation
│   ├── main.py              # Streamlit web interface
│   ├── config/
│   │   ├── agents.yaml      # Agent configurations
│   │   └── tasks.yaml       # Task configurations
│   └── tools/
│       └── milvus.py        # Milvus integration tools
├── launch_crew.py           # Test launcher
├── run_crew_example.py      # Usage examples
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔍 Testing

Run the comprehensive test suite:
```bash
python src/crew/test_gemini_integration.py
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the documentation
2. Run the test script to verify your setup
3. Review the example usage files
4. Check environment variable configuration

## 🔮 Future Enhancements

- [ ] Integration with Electronic Health Records (EHR)
- [ ] Multi-language medical document support
- [ ] Advanced medical image analysis
- [ ] Real-time medical monitoring
- [ ] Integration with medical APIs and databases
- [ ] Mobile application interface
