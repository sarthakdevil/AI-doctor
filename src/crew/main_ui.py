__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.modules["sqlite3.dbapi2"] = sys.modules["pysqlite3.dbapi2"]
#!/usr/bin/env python3
"""
AI Doctor Crew - Streamlit Web Interface
Comprehensive medical document processing and analysis system
"""

import streamlit as st
import os
import tempfile
import json
import time
import traceback
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
import sys

# Add the current directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our crew and tools
try:
    from doc_crew import DocCrew
    from tools.milvus import ChromaDBClient
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    # Don't stop here, let the UI handle it gracefully
    DocCrew = None
    ChromaDBClient = None

# Page configuration
st.set_page_config(
    page_title="AI Doctor Chat 🏥",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'doc_crew' not in st.session_state:
        st.session_state.doc_crew = None
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'system_status' not in st.session_state:
        st.session_state.system_status = "Not Initialized"
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}
    if 'medical_context' not in st.session_state:
        st.session_state.medical_context = ""
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = "initial"  # initial, general, documents
    if 'show_api_config' not in st.session_state:
        st.session_state.show_api_config = False

def check_environment_setup():
    """Check if all required environment variables are set"""
    required_vars = ["OPENAI_API_KEY", "MILVUS_URI", "MILVUS_TOKEN", "BYTEZ_API_KEY", "GEMINI_API_KEY", "SERPER_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    return missing_vars

def configure_api_keys():
    """UI component to configure API keys"""
    st.markdown("### 🔑 API Keys Configuration")
    st.info("Please enter your API keys below. They will be saved for this session.")
    
    # Default values for demo/testing
    default_values = {
        "MILVUS_URI": "https://in03-5a8809cb4165fba.serverless.gcp-us-west1.cloud.zilliz.com",
        "MILVUS_TOKEN": "",  # Will be constructed from user/password
        "BYTEZ_API_KEY": "008500fe8f2fd1acbdd6ae89df607343",
        "GEMINI_API_KEY": "AIzaSyC-QKbC721mwx0c9OWAVxzRZkaqtbi8-YA",
        "SERPER_API_KEY": "c1ef6e40d489b6728585aebcc6b832a7384627a1",
        "OPENAI_API_KEY": ""
    }
    
    with st.form("api_keys_form"):
        st.markdown("#### Required API Keys:")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            placeholder="sk-..."
        )
        
        # Milvus Configuration
        st.markdown("#### Milvus Database Configuration:")
        milvus_user = st.text_input(
            "Milvus Username", 
            value=os.getenv("MILVUS_USER", "db_5a8809cb4165fba"),
            placeholder="db_5a8809cb4165fba"
        )
        milvus_password = st.text_input(
            "Milvus Password", 
            value=os.getenv("MILVUS_PASSWORD", "Tu3!zF5t%nS3>][9"),
            type="password",
            placeholder="Your Milvus password"
        )
        milvus_url = st.text_input(
            "Milvus URL", 
            value=os.getenv("MILVUS_URI", default_values["MILVUS_URI"]),
            placeholder="https://..."
        )
        
        # Other API Keys
        st.markdown("#### Additional API Keys:")
        bytez_key = st.text_input(
            "Bytez API Key", 
            value=os.getenv("BYTEZ_API_KEY", default_values["BYTEZ_API_KEY"]),
            type="password"
        )
        gemini_key = st.text_input(
            "Gemini API Key", 
            value=os.getenv("GEMINI_API_KEY", default_values["GEMINI_API_KEY"]),
            type="password"
        )
        serper_key = st.text_input(
            "Serper API Key", 
            value=os.getenv("SERPER_API_KEY", default_values["SERPER_API_KEY"]),
            type="password"
        )
        
        submitted = st.form_submit_button("💾 Save API Keys", type="primary")
        
        if submitted:
            # Save to environment
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if milvus_user:
                os.environ["MILVUS_USER"] = milvus_user
            if milvus_password:
                os.environ["MILVUS_PASSWORD"] = milvus_password
            if milvus_url:
                os.environ["MILVUS_URI"] = milvus_url
                # Create Milvus token from user:password
                token = base64.b64encode(f"{milvus_user}:{milvus_password}".encode()).decode()
                os.environ["MILVUS_TOKEN"] = token
            if bytez_key:
                os.environ["BYTEZ_API_KEY"] = bytez_key
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
            if serper_key:
                os.environ["SERPER_API_KEY"] = serper_key
            
            st.success("✅ API keys saved successfully!")
            st.info("🔄 Please refresh the page or reinitialize the system to apply changes.")
            
            # Show which keys are now set
            missing_vars = check_environment_setup()
            if missing_vars:
                st.warning(f"⚠️ Still missing: {', '.join(missing_vars)}")
            else:
                st.success("🎉 All API keys are now configured!")
            
            time.sleep(2)
            st.rerun()
    
    # Quick setup button with demo credentials
    st.markdown("---")
    st.markdown("#### 🚀 Quick Demo Setup")
    st.info("Use pre-configured demo credentials for testing (not recommended for production)")
    
    if st.button("🎯 Setup Demo Environment", type="secondary"):
        # Set demo environment
        os.environ["MILVUS_USER"] = "db_5a8809cb4165fba"
        os.environ["MILVUS_PASSWORD"] = "Tu3!zF5t%nS3>][9"
        os.environ["MILVUS_URI"] = default_values["MILVUS_URI"]
        os.environ["BYTEZ_API_KEY"] = default_values["BYTEZ_API_KEY"]
        os.environ["GEMINI_API_KEY"] = default_values["GEMINI_API_KEY"]
        os.environ["SERPER_API_KEY"] = default_values["SERPER_API_KEY"]
        
        # Create Milvus token
        token = base64.b64encode("db_5a8809cb4165fba:Tu3!zF5t%nS3>][9".encode()).decode()
        os.environ["MILVUS_TOKEN"] = token
        
        st.success("✅ Demo environment configured! (You still need to add your OpenAI API key)")
        st.rerun()

def initialize_crew():
    """Initialize the DocCrew system"""
    # Check if imports were successful
    if not IMPORTS_SUCCESSFUL:
        st.error(f"❌ Import error: {IMPORT_ERROR}")
        
        # Provide specific help for PyMuPDF issues
        if "fitz" in IMPORT_ERROR.lower() or "pymupdf" in IMPORT_ERROR.lower():
            st.error("🔧 **PyMuPDF Installation Issue**")
            st.code("pip install PyMuPDF", language="bash")
            st.info("PyMuPDF is required for PDF text extraction. Please install it and restart the application.")
        
        # Provide general installation help
        st.subheader("📦 Installation Steps:")
        st.code("""
# Install all required packages
cd "c:/Users/sarth/OneDrive/Desktop/kartavya-internship/AI-doctor/crew"
pip install -e .
        """, language="bash")
        
        return False
    
    try:
        if st.session_state.doc_crew is None:
            with st.spinner("Initializing AI Doctor Crew..."):
                st.session_state.doc_crew = DocCrew()
                st.session_state.system_status = "Ready"
            st.success("✅ AI Doctor Crew initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize crew: {str(e)}")
        st.session_state.system_status = f"Error: {str(e)}"
        
        # Provide helpful error information
        if "fitz" in str(e).lower() or "pymupdf" in str(e).lower():
            st.error("🔧 **PyMuPDF Issue Detected**")
            st.info("This is typically resolved by reinstalling PyMuPDF:")
            st.code("pip uninstall PyMuPDF && pip install PyMuPDF", language="bash")
        
        return False

def main_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🩺 AI Doctor Chat - Your Personal Medical Assistant</h1>
    </div>
    """, unsafe_allow_html=True)

def sidebar_status():
    """Display system status in sidebar"""
    st.sidebar.title("📊 System Status")
    
    # Environment check
    missing_vars = check_environment_setup()
    if missing_vars:
        st.sidebar.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        st.sidebar.info("Please set up your .env file with required variables")
    else:
        st.sidebar.success("✅ Environment configured")
    
    # Crew status
    status_color = "🟢" if st.session_state.system_status == "Ready" else "🔴"
    st.sidebar.write(f"**Crew Status:** {status_color} {st.session_state.system_status}")
    
    # Statistics
    st.sidebar.write("**Statistics:**")
    st.sidebar.write(f"📄 Documents Uploaded: {len(st.session_state.processed_documents)}")
    st.sidebar.write(f"� Chat Messages: {len(st.session_state.chat_history)}")
    
    # Quick actions
    st.sidebar.title("⚡ Quick Actions")
    if st.sidebar.button("� Configure API Keys"):
        st.session_state.show_api_config = True
        st.rerun()
    
    if st.sidebar.button("�🔄 Reinitialize System"):
        st.session_state.doc_crew = None
        st.session_state.system_status = "Not Initialized"
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.chat_mode = "initial"
        st.success("Chat history cleared!")
    
    # Debug info
    with st.sidebar.expander("🔧 Debug Info"):
        st.write(f"Chat Mode: {st.session_state.chat_mode}")
        st.write(f"Profile Keys: {list(st.session_state.user_profile.keys()) if st.session_state.user_profile else 'None'}")
        st.write(f"Chat Messages: {len(st.session_state.chat_history)}")
        st.write(f"Medical Context: {st.session_state.medical_context[:50]}..." if st.session_state.medical_context else "None")

def document_processing_tab():
    """Document Processing Tab"""
    st.header("📄 Document Processing")
    
    # Upload section
    st.subheader("📄 Upload Medical Documents")
    st.info("📝 **Note:** Uploading documents will create vector embeddings for future analysis. The AI crew will run when you ask questions.")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload medical documents, research papers, clinical guidelines, etc. These will be processed into searchable embeddings."
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Files selected:** {len(uploaded_files)}")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        with col2:
            if st.button("� Create Embeddings", type="primary"):
                process_documents(uploaded_files)
    
    # Processing history
    st.subheader("📚 Document Embeddings Status")
    if st.session_state.processed_documents:
        df = pd.DataFrame(st.session_state.processed_documents)
        st.dataframe(df, use_container_width=True)
        st.info("💡 **Ready for Analysis:** Your documents are now searchable. Go to the Medical Analysis tab to ask questions!")
    else:
        st.info("No documents processed yet. Upload some medical PDFs to create searchable embeddings!")

def process_documents(uploaded_files):
    """Process uploaded documents and create vector embeddings only"""
    if not st.session_state.doc_crew:
        st.error("❌ Please initialize the system first")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize ChromaDBClient for direct document processing
    try:
        chromadb_client = ChromaDBClient()
    except Exception as e:
        st.error(f"❌ Failed to initialize ChromaDB client: {str(e)}")
        return
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Creating embeddings for {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process PDF and create documents only (no crew execution)
            success = chromadb_client.process_pdf_and_save(
                pdf_path=tmp_file_path,
                chunk_size=500,
                overlap=50,
                custom_metadata={
                    'upload_source': 'web_ui',
                    'original_filename': uploaded_file.name
                }
            )
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Store processing info
            if success:
                doc_info = {
                    "filename": uploaded_file.name,
                    "size_kb": uploaded_file.size / 1024,
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Embeddings Created",
                    "result": "Vector embeddings successfully created and stored in database"
                }
                st.session_state.processed_documents.append(doc_info)
            else:
                raise Exception("Failed to create embeddings")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            doc_info = {
                "filename": uploaded_file.name,
                "size_kb": uploaded_file.size / 1024,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Error",
                "result": str(e)
            }
            st.session_state.processed_documents.append(doc_info)
    
    status_text.text("✅ All documents processed and embeddings created!")
    st.success(f"Successfully created embeddings for {len(uploaded_files)} documents!")

def medical_analysis_tab():
    """Medical Analysis Tab"""
    st.header("🔍 Medical Analysis")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "General Medical Query",
            "Symptom Analysis", 
            "Drug Interaction Check",
            "Treatment Protocol Research",
            "Medical Literature Review"
        ]
    )
    
    # Input based on analysis type
    if analysis_type == "General Medical Query":
        medical_query_analysis()
    elif analysis_type == "Symptom Analysis":
        symptom_analysis()
    elif analysis_type == "Drug Interaction Check":
        drug_interaction_analysis()
    elif analysis_type == "Treatment Protocol Research":
        treatment_protocol_analysis()
    elif analysis_type == "Medical Literature Review":
        literature_review_analysis()

def medical_query_analysis():
    """General medical query analysis"""
    st.subheader("🩺 General Medical Query Analysis")
    
    # Query input
    query = st.text_area(
        "Enter your medical question or query:",
        height=100,
        placeholder="e.g., What are the diagnostic criteria for Type 2 diabetes?"
    )
    
    # Additional context
    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input("Patient Age (optional)", min_value=0, max_value=120, value=0)
        patient_gender = st.selectbox("Patient Gender (optional)", ["Not specified", "Male", "Female", "Other"])
    
    with col2:
        urgency = st.selectbox("Urgency Level", ["Low", "Medium", "High", "Emergency"])
        specialty = st.selectbox("Medical Specialty", [
            "General Medicine", "Cardiology", "Neurology", "Oncology", 
            "Pediatrics", "Psychiatry", "Surgery", "Emergency Medicine"
        ])
    
    if st.button("🔍 Analyze Query", type="primary"):
        if query.strip():
            run_medical_analysis(query, {
                "age": patient_age if patient_age > 0 else None,
                "gender": patient_gender if patient_gender != "Not specified" else None,
                "urgency": urgency,
                "specialty": specialty
            })
        else:
            st.warning("Please enter a medical query")

def symptom_analysis():
    """Symptom analysis interface"""
    st.subheader("🩺 Symptom Analysis")
    
    # Symptom input
    symptoms = st.text_area(
        "Describe the symptoms:",
        height=120,
        placeholder="e.g., Patient presents with chest pain, shortness of breath, and fatigue for the past 3 days..."
    )
    
    # Symptom details
    col1, col2, col3 = st.columns(3)
    with col1:
        duration = st.selectbox("Duration", ["Acute (< 24 hours)", "Subacute (1-7 days)", "Chronic (> 1 week)"])
        severity = st.slider("Severity (1-10)", 1, 10, 5)
    
    with col2:
        onset = st.selectbox("Onset", ["Sudden", "Gradual", "Progressive"])
        location = st.text_input("Primary Location", placeholder="e.g., chest, abdomen, head")
    
    with col3:
        triggers = st.text_input("Triggers/Aggravating factors", placeholder="e.g., exercise, stress")
        relief = st.text_input("Relieving factors", placeholder="e.g., rest, medication")
    
    if st.button("🔍 Analyze Symptoms", type="primary"):
        if symptoms.strip():
            symptom_context = {
                "symptoms": symptoms,
                "duration": duration,
                "severity": severity,
                "onset": onset,
                "location": location,
                "triggers": triggers,
                "relief": relief
            }
            run_symptom_analysis(symptom_context)
        else:
            st.warning("Please describe the symptoms")

def drug_interaction_analysis():
    """Drug interaction analysis interface"""
    st.subheader("💊 Drug Interaction Check")
    
    # Medication input
    medications = st.text_area(
        "Enter current medications:",
        height=120,
        placeholder="e.g., Warfarin 5mg daily, Aspirin 81mg daily, Metoprolol 50mg twice daily"
    )
    
    # Additional medication details
    col1, col2 = st.columns(2)
    with col1:
        allergies = st.text_area("Known allergies:", placeholder="e.g., Penicillin, Sulfa drugs")
        medical_conditions = st.text_area("Medical conditions:", placeholder="e.g., Hypertension, Diabetes")
    
    with col2:
        new_medication = st.text_input("New medication to check:", placeholder="e.g., Ibuprofen")
        check_type = st.selectbox("Check Type", [
            "Drug-Drug Interactions",
            "Drug-Food Interactions", 
            "Drug-Condition Interactions",
            "Comprehensive Analysis"
        ])
    
    if st.button("💊 Check Interactions", type="primary"):
        if medications.strip():
            drug_context = {
                "medications": medications,
                "allergies": allergies,
                "medical_conditions": medical_conditions,
                "new_medication": new_medication,
                "check_type": check_type
            }
            run_drug_interaction_check(drug_context)
        else:
            st.warning("Please enter current medications")

def treatment_protocol_analysis():
    """Treatment protocol research interface"""
    st.subheader("📋 Treatment Protocol Research")
    
    condition = st.text_input(
        "Medical Condition:",
        placeholder="e.g., Acute Myocardial Infarction, Type 2 Diabetes"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        guidelines = st.multiselect("Preferred Guidelines", [
            "AHA/ACC", "ESC", "ADA", "WHO", "NICE", "ACCP", "ASCO", "NCCN"
        ])
        evidence_level = st.selectbox("Evidence Level", ["All", "Level A", "Level B", "Level C"])
    
    with col2:
        population = st.selectbox("Patient Population", [
            "Adult", "Pediatric", "Elderly", "Pregnant", "All populations"
        ])
        setting = st.selectbox("Clinical Setting", [
            "Outpatient", "Inpatient", "Emergency", "ICU", "All settings"
        ])
    
    if st.button("📋 Research Protocols", type="primary"):
        if condition.strip():
            protocol_context = {
                "condition": condition,
                "guidelines": guidelines,
                "evidence_level": evidence_level,
                "population": population,
                "setting": setting
            }
            run_treatment_protocol_research(protocol_context)
        else:
            st.warning("Please enter a medical condition")

def literature_review_analysis():
    """Medical literature review interface"""
    st.subheader("📚 Medical Literature Review")
    
    topic = st.text_input(
        "Research Topic:",
        placeholder="e.g., Efficacy of statins in primary prevention of cardiovascular disease"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        study_types = st.multiselect("Study Types", [
            "Randomized Controlled Trials", "Meta-analyses", "Systematic Reviews",
            "Cohort Studies", "Case-Control Studies", "Case Reports"
        ])
        time_period = st.selectbox("Time Period", [
            "Last 1 year", "Last 3 years", "Last 5 years", "Last 10 years", "All time"
        ])
    
    with col2:
        population_focus = st.text_input("Population Focus", placeholder="e.g., adults, children, elderly")
        outcome_measures = st.text_area("Outcome Measures", placeholder="e.g., mortality, morbidity, quality of life")
    
    if st.button("📚 Review Literature", type="primary"):
        if topic.strip():
            review_context = {
                "topic": topic,
                "study_types": study_types,
                "time_period": time_period,
                "population_focus": population_focus,
                "outcome_measures": outcome_measures
            }
            run_literature_review(review_context)
        else:
            st.warning("Please enter a research topic")

def run_medical_analysis(query: str, context: Dict[str, Any]):
    """Run general medical analysis"""
    if not st.session_state.doc_crew:
        st.error("❌ Please initialize the system first")
        return
    
    with st.spinner("🔍 Analyzing medical query..."):
        try:
            # Format query with context
            formatted_query = f"Medical Query: {query}\n"
            if context.get('age'):
                formatted_query += f"Patient Age: {context['age']}\n"
            if context.get('gender'):
                formatted_query += f"Patient Gender: {context['gender']}\n"
            formatted_query += f"Urgency: {context['urgency']}\n"
            formatted_query += f"Specialty: {context['specialty']}"
            
            # Use crew.kickoff() with single user_query input
            result = st.session_state.doc_crew.crew().kickoff(inputs={"user_query": formatted_query})
            
            # Display results
            display_analysis_results("Medical Analysis", query, result, context)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.error("Error details:", traceback.format_exc())

def run_symptom_analysis(context: Dict[str, Any]):
    """Run symptom analysis"""
    if not st.session_state.doc_crew:
        st.error("❌ Please initialize the system first")
        return
    
    with st.spinner("🩺 Analyzing symptoms..."):
        try:
            formatted_symptoms = f"""
            Symptoms: {context['symptoms']}
            Duration: {context['duration']}
            Severity: {context['severity']}/10
            Onset: {context['onset']}
            Location: {context['location']}
            Triggers: {context['triggers']}
            Relieving factors: {context['relief']}
            """
            
            # Use crew.kickoff() with single user_query input
            result = st.session_state.doc_crew.crew().kickoff(inputs={"user_query": formatted_symptoms})
            
            # Display results
            display_analysis_results("Symptom Analysis", context['symptoms'], result, context)
            
        except Exception as e:
            st.error(f"❌ Symptom analysis failed: {str(e)}")

def run_drug_interaction_check(context: Dict[str, Any]):
    """Run drug interaction check"""
    if not st.session_state.doc_crew:
        st.error("❌ Please initialize the system first")
        return
    
    with st.spinner("💊 Checking drug interactions..."):
        try:
            formatted_request = f"""
            Current Medications: {context['medications']}
            Allergies: {context['allergies']}
            Medical Conditions: {context['medical_conditions']}
            New Medication: {context['new_medication']}
            Check Type: {context['check_type']}
            """
            
            # Use crew.kickoff() with single user_query input
            result = st.session_state.doc_crew.crew().kickoff(inputs={"user_query": formatted_request})
            
            # Display results
            display_analysis_results("Drug Interaction Check", context['medications'], result, context)
            
        except Exception as e:
            st.error(f"❌ Drug interaction check failed: {str(e)}")

def run_treatment_protocol_research(context: Dict[str, Any]):
    """Run treatment protocol research"""
    if not st.session_state.doc_crew:
        st.error("❌ Please initialize the system first")
        return
    
    with st.spinner("📋 Researching treatment protocols..."):
        try:
            formatted_request = f"""
            Research treatment protocols for: {context['condition']}
            Guidelines: {', '.join(context['guidelines']) if context['guidelines'] else 'All'}
            Evidence Level: {context['evidence_level']}
            Population: {context['population']}
            Setting: {context['setting']}
            """
            
            # Use crew.kickoff() with single user_query input
            result = st.session_state.doc_crew.crew().kickoff(inputs={"user_query": formatted_request})
            
            # Display results
            display_analysis_results("Treatment Protocol Research", context['condition'], result, context)
            
        except Exception as e:
            st.error(f"❌ Treatment protocol research failed: {str(e)}")

def run_literature_review(context: Dict[str, Any]):
    """Run literature review"""
    if not st.session_state.doc_crew:
        st.error("❌ Please initialize the system first")
        return
    
    with st.spinner("📚 Reviewing medical literature..."):
        try:
            formatted_request = f"""
            Literature review topic: {context['topic']}
            Study types: {', '.join(context['study_types']) if context['study_types'] else 'All'}
            Time period: {context['time_period']}
            Population focus: {context['population_focus']}
            Outcome measures: {context['outcome_measures']}
            """
            
            # Use crew.kickoff() with single user_query input
            result = st.session_state.doc_crew.crew().kickoff(inputs={"user_query": formatted_request})
            
            # Display results
            display_analysis_results("Literature Review", context['topic'], result, context)
            
        except Exception as e:
            st.error(f"❌ Literature review failed: {str(e)}")

def display_analysis_results(analysis_type: str, query: str, result: Any, context: Dict[str, Any]):
    """Display analysis results"""
    st.success("✅ Analysis completed!")
    
    # Create expandable result section
    with st.expander("📋 Analysis Results", expanded=True):
        st.markdown("### 🔍 Analysis Summary")
        st.write(f"**Type:** {analysis_type}")
        st.write(f"**Query:** {query[:100]}..." if len(query) > 100 else f"**Query:** {query}")
        st.write(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("### 📊 Results")
        if isinstance(result, str):
            st.markdown(result)
        else:
            st.json(result)
    
    # Medical disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong><br>
        This analysis is for educational and research purposes only. It should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
        professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Save to history
    analysis_record = {
        "type": analysis_type,
        "query": query,
        "result": str(result)[:500] + "..." if len(str(result)) > 500 else str(result),
        "context": context,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.analysis_history.append(analysis_record)

def analysis_history_tab():
    """Analysis History Tab"""
    st.header("📊 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analyses completed yet. Start with document processing or medical queries!")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        analysis_types = [record['type'] for record in st.session_state.analysis_history]
        selected_type = st.selectbox("Filter by Type", ["All"] + list(set(analysis_types)))
    
    with col2:
        time_filter = st.selectbox("Time Period", ["All", "Today", "This Week", "This Month"])
    
    with col3:
        if st.button("📥 Export History"):
            export_history()
    
    # Display history
    filtered_history = st.session_state.analysis_history
    if selected_type != "All":
        filtered_history = [r for r in filtered_history if r['type'] == selected_type]
    
    for i, record in enumerate(reversed(filtered_history)):
        with st.expander(f"🔍 {record['type']} - {record['timestamp']}"):
            st.write(f"**Query:** {record['query']}")
            st.write(f"**Result Preview:** {record['result'][:200]}...")
            if st.button(f"View Full Result", key=f"view_{i}"):
                st.markdown("#### Full Result:")
                st.markdown(record['result'])

def export_history():
    """Export analysis history"""
    if st.session_state.analysis_history:
        df = pd.DataFrame(st.session_state.analysis_history)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"ai_doctor_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No history to export")

def system_settings_tab():
    """System Settings Tab"""
    st.header("⚙️ System Settings")
    
    # Import status check
    st.subheader("📦 Dependency Status")
    
    if not IMPORTS_SUCCESSFUL:
        st.error(f"❌ Import Error: {IMPORT_ERROR}")
        
        # Show specific installation commands
        st.subheader("🔧 Fix Installation Issues")
        
        if "fitz" in IMPORT_ERROR.lower() or "pymupdf" in IMPORT_ERROR.lower():
            st.error("**PyMuPDF Issue Detected**")
            st.code("pip install --upgrade PyMuPDF", language="bash")
            st.info("You may need to restart the application after installation")
            
        st.subheader("💾 Complete Reinstallation")
        st.code("""
# Reinstall all dependencies
cd "c:/Users/sarth/OneDrive/Desktop/kartavya-internship/AI-doctor/crew"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
        """, language="bash")
        
        # Run dependency check button
        if st.button("🔍 Run Dependency Check"):
            with st.spinner("Checking dependencies..."):
                # This will run our dependency check
                st.code("python install_dependencies.py", language="bash")
                st.info("Please run this command in your terminal to check and install dependencies")
    else:
        st.success("✅ All core dependencies imported successfully")
        
        # Show Python environment info
        st.subheader("🐍 Python Environment")
        import sys
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Python Executable:** {sys.executable}")
        
        # Test key modules
        modules_status = {}
        test_modules = [
            ('fitz', 'PyMuPDF'),
            ('crewai', 'CrewAI'),
            ('streamlit', 'Streamlit'),
            ('pymilvus', 'Milvus'),
            ('google.generativeai', 'Google AI')
        ]
        
        for module, display_name in test_modules:
            try:
                imported = __import__(module)
                if hasattr(imported, '__version__'):
                    version = imported.__version__
                elif hasattr(imported, 'version'):
                    version = imported.version
                else:
                    version = "Unknown"
                modules_status[display_name] = f"✅ {version}"
            except ImportError:
                modules_status[display_name] = "❌ Not Available"
        
        st.subheader("📚 Module Versions")
        for module, status in modules_status.items():
            st.write(f"**{module}:** {status}")
    
    # Environment variables
    st.subheader("🔧 Environment Configuration")
    
    missing_vars = check_environment_setup()
    if missing_vars:
        st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        
        # Show API key configuration in settings
        with st.expander("🔑 Configure API Keys", expanded=True):
            configure_api_keys()
    else:
        st.success("✅ All environment variables configured")
        
        # Show current configuration (masked)
        with st.expander("🔍 View Current Configuration"):
            config_info = {
                "OpenAI API Key": "sk-..." + os.getenv("OPENAI_API_KEY", "")[-4:] if os.getenv("OPENAI_API_KEY") else "Not set",
                "Milvus URI": os.getenv("MILVUS_URI", "Not set"),
                "Milvus User": os.getenv("MILVUS_USER", "Not set"),
                "Bytez API Key": "***" + os.getenv("BYTEZ_API_KEY", "")[-4:] if os.getenv("BYTEZ_API_KEY") else "Not set",
                "Gemini API Key": "***" + os.getenv("GEMINI_API_KEY", "")[-4:] if os.getenv("GEMINI_API_KEY") else "Not set",
                "Serper API Key": "***" + os.getenv("SERPER_API_KEY", "")[-4:] if os.getenv("SERPER_API_KEY") else "Not set"
            }
            
            for key, value in config_info.items():
                st.write(f"**{key}:** {value}")
            
            if st.button("🔄 Reconfigure API Keys"):
                st.session_state.show_api_config = True
                st.rerun()
    
    # System controls
    st.subheader("🎛️ System Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Restart System", type="secondary"):
            st.session_state.doc_crew = None
            st.session_state.system_status = "Not Initialized"
            st.success("System restarted. Please reinitialize.")
    
    with col2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will clear all data"):
                st.session_state.processed_documents = []
                st.session_state.analysis_history = []
                st.success("All data cleared!")
    
    # System information
    st.subheader("ℹ️ System Information")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.metric("Documents Processed", len(st.session_state.processed_documents))
        st.metric("Analyses Completed", len(st.session_state.analysis_history))
    
    with info_col2:
        st.metric("System Status", st.session_state.system_status)
        st.metric("Import Status", "✅ Success" if IMPORTS_SUCCESSFUL else "❌ Failed")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    main_header()
    
    # Sidebar status
    sidebar_status()
    
    # Show API key configuration if requested or if keys are missing
    missing_vars = check_environment_setup()
    if st.session_state.show_api_config or missing_vars:
        if missing_vars:
            st.warning(f"⚠️ Some environment variables are missing: {', '.join(missing_vars)}")
        
        # Show API key configuration UI
        configure_api_keys()
        
        # Reset the flag after showing
        if st.session_state.show_api_config:
            st.session_state.show_api_config = False
        
        # Don't continue if critical keys are missing
        if "OPENAI_API_KEY" in missing_vars:
            st.error("❌ OpenAI API Key is required for AI functionality. Please configure it above.")
            return
    
    # Try to continue with available functionality
    if missing_vars:
        st.info("💡 Some features may be limited due to missing API keys.")
    
    # Initialize crew
    if not st.session_state.doc_crew:
        try:
            with st.spinner("Initializing AI Doctor..."):
                st.session_state.doc_crew = DocCrew()
                if missing_vars:
                    st.session_state.system_status = "Ready (Limited)"
                else:
                    st.session_state.system_status = "Ready"
            st.success("✅ AI Doctor initialized successfully!")
        except Exception as e:
            st.error(f"❌ Could not initialize AI Doctor: {str(e)}")
            st.info("💡 Please check your API keys and try again.")
            return
    
    # Main chat interface
    chat_interface()

def chat_interface():
    """Main chat interface"""
    st.markdown("### 👋 Welcome to AI Doctor Chat!")
    
    # Check if user has uploaded documents or chosen general chat
    if st.session_state.chat_mode == "initial":
        upload_medical_documents()
    else:
        # Show chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 💬 Chat with AI Doctor")
            
            # Display chat history
            display_chat_history()
            
            # Chat input
            user_message = st.chat_input("Ask me anything about your health, symptoms, or medical concerns...")
            
            if user_message:
                handle_user_message(user_message)
        
        with col2:
            # Show uploaded documents and user profile
            show_medical_context()
            
            # Option to upload more documents
            if st.session_state.chat_mode == "general":
                st.markdown("---")
                st.markdown("#### 📄 Upload Documents")
                st.info("You can upload medical documents anytime to get more personalized advice.")
                if st.button("+ Upload Medical Records"):
                    upload_additional_documents()
            else:
                st.markdown("---")
                st.markdown("#### 📄 Upload More Documents")
                if st.button("+ Add More Medical Records"):
                    upload_additional_documents()

def upload_medical_documents():
    """Handle initial medical document upload"""
    st.markdown("""
    #### 📋 Upload Your Medical History
    
    Please upload your medical reports, test results, or any relevant medical documents. 
    This will help me provide better and more personalized medical advice.
    
    **Or you can skip this step and start chatting directly for general medical questions.**
    """)
    
    # Initialize profile variables with defaults
    age = 30
    gender = "Not specified"
    height = 170
    weight = 70
    medical_conditions = ""
    current_medications = ""
    
    # User profile section
    with st.expander("👤 Tell me about yourself (Optional)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
            
        with col2:
            height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
        
        medical_conditions = st.text_area(
            "Known medical conditions (if any)",
            placeholder="e.g., Diabetes, Hypertension, Allergies..."
        )
        
        current_medications = st.text_area(
            "Current medications (if any)",
            placeholder="e.g., Metformin 500mg twice daily..."
        )
    
    # Skip option prominently displayed
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("💬 Skip & Start General Chat", type="primary", use_container_width=True):
            # Save basic profile if provided (preserve user input even when skipping)
            st.session_state.user_profile = {
                "age": age if age > 0 else None,
                "gender": gender if gender != "Not specified" else None,
                "height": height if height != 170 else None,
                "weight": weight if weight != 70 else None,
                "medical_conditions": medical_conditions if medical_conditions.strip() else None,
                "current_medications": current_medications if current_medications.strip() else None
            }
            st.session_state.medical_context = "No medical documents uploaded - providing general medical assistance."
            st.session_state.chat_mode = "general"
            
            # Add welcome message for general chat
            profile_info = ""
            if any(v for v in st.session_state.user_profile.values() if v is not None):
                profile_info = f"""
                
**Your Profile:** 
- Age: {st.session_state.user_profile.get('age', 'Not specified')}
- Gender: {st.session_state.user_profile.get('gender', 'Not specified')}
- Medical Conditions: {st.session_state.user_profile.get('medical_conditions', 'None specified')}
- Current Medications: {st.session_state.user_profile.get('current_medications', 'None specified')}
                """
            
            welcome_message = f"""
            👋 Hello! I'm your AI Medical Assistant ready to help with general medical questions.{profile_info}
            
            I can help you with:
            - General medical questions
            - Symptom information and education
            - Health education
            - Medical terminology explanations
            - General wellness advice
            
            **Note:** Since you haven't uploaded medical documents, I'll provide general medical information. For personalized advice, please consult with your healthcare provider.
            
            What would you like to know about today?
            """
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": welcome_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
    
    with col2:
        st.markdown("**OR**")
    
    # File upload section
    st.markdown("#### 📁 Upload Medical Documents")
    uploaded_files = st.file_uploader(
        "Choose your medical files (PDF format)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload medical reports, lab results, prescriptions, or any relevant medical documents"
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Files selected:** {len(uploaded_files)}")
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
        
        with col2:
            if st.button("🚀 Process Documents & Start Chat", type="primary"):
                # Save user profile
                st.session_state.user_profile = {
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "medical_conditions": medical_conditions,
                    "current_medications": current_medications
                }
                
                # Set mode to documents
                st.session_state.chat_mode = "documents"
                
                # Process documents
                process_documents_for_chat(uploaded_files)
    
    else:
        st.info("👆 Upload your medical documents for personalized advice, or use the 'Skip & Start General Chat' button above for general medical questions.")
        
        # Additional skip option at the bottom
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("💬 Start General Medical Chat", use_container_width=True):
                # Save basic profile if provided
                st.session_state.user_profile = {
                    "age": age if age > 0 else None,
                    "gender": gender if gender != "Not specified" else None,
                    "height": height if height != 170 else None,
                    "weight": weight if weight != 70 else None,
                    "medical_conditions": medical_conditions if medical_conditions.strip() else None,
                    "current_medications": current_medications if current_medications.strip() else None
                }
                st.session_state.medical_context = "General medical chat mode - no documents uploaded."
                st.session_state.chat_mode = "general"
                
                # Add welcome message
                profile_info = ""
                if any(v for v in st.session_state.user_profile.values() if v is not None):
                    profile_info = f"""
                    
**Your Profile:** 
- Age: {st.session_state.user_profile.get('age', 'Not specified')}
- Gender: {st.session_state.user_profile.get('gender', 'Not specified')}
- Medical Conditions: {st.session_state.user_profile.get('medical_conditions', 'None specified')}
- Current Medications: {st.session_state.user_profile.get('current_medications', 'None specified')}
                    """
                
                welcome_message = f"""
                👋 Hello! I'm your AI Medical Assistant ready to help with general medical questions.{profile_info}
                
                I can assist you with:
                - General health questions
                - Symptom information and education
                - Medical terminology explanations
                - Wellness and prevention advice
                - Medication information
                - Health condition overviews
                
                **Important:** This is for general information only. For personalized medical advice, diagnosis, or treatment, please consult with qualified healthcare professionals.
                
                What would you like to know about today?
                """
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": welcome_message,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()

def process_documents_for_chat(uploaded_files):
    """Process uploaded documents for chat context - create documents only"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize ChromaDBClient for direct document processing
    try:
        chromadb_client = ChromaDBClient()
    except Exception as e:
        st.error(f"❌ Failed to initialize ChromaDB client: {str(e)}")
        return
    
    medical_context_parts = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Creating embeddings for {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process PDF and create documents only (no crew execution)
            success = chromadb_client.process_pdf_and_save(
                pdf_path=tmp_file_path,
                chunk_size=500,
                overlap=50,
                custom_metadata={
                    'upload_source': 'chat_interface',
                    'original_filename': uploaded_file.name,
                    'user_profile': str(st.session_state.user_profile)
                }
            )
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Store processing info
            if success:
                doc_info = {
                    "filename": uploaded_file.name,
                    "size_kb": uploaded_file.size / 1024,
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Embeddings Created",
                    "content_summary": "Vector embeddings successfully created and stored in database"
                }
                st.session_state.processed_documents.append(doc_info)
                medical_context_parts.append(f"Document: {uploaded_file.name} - Embeddings created successfully")
            else:
                raise Exception("Failed to create embeddings")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            doc_info = {
                "filename": uploaded_file.name,
                "size_kb": uploaded_file.size / 1024,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Error",
                "content_summary": str(e)
            }
            st.session_state.processed_documents.append(doc_info)
    
    # Create consolidated medical context
    st.session_state.medical_context = "\n\n".join(medical_context_parts)
    st.session_state.chat_mode = "documents"
    
    status_text.text("✅ Document embeddings created successfully!")
    st.success("🎉 Great! Your medical documents have been processed into searchable embeddings. You can now start chatting with the AI Doctor!")
    
    # Add welcome message to chat
    welcome_message = f"""
    👋 Hello! I've processed your medical documents into searchable embeddings. Here's what I know about you:
    
    **Profile:** {st.session_state.user_profile.get('age', 'N/A')} years old, {st.session_state.user_profile.get('gender', 'N/A')}
    **Medical Conditions:** {st.session_state.user_profile.get('medical_conditions', 'None specified')}
    **Current Medications:** {st.session_state.user_profile.get('current_medications', 'None specified')}
    **Documents Processed:** {len(st.session_state.processed_documents)} medical documents into searchable embeddings
    
    Feel free to ask me any questions about your health, symptoms, medications, or anything related to your medical care. I'm here to help! 
    
    💡 You can ask things like:
    - "What do my test results mean?"
    - "Are there any concerning findings?"
    - "What should I know about my current medications?"
    - "I have [symptoms], what could this mean?"
    """
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": welcome_message,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    time.sleep(2)
    st.rerun()

def display_chat_history():
    """Display the chat history"""
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>👋 Start a conversation!</h3>
            <p>Ask me anything about your health or medical concerns.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display messages
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**You** - {message['timestamp']}")
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**AI Doctor** - {message['timestamp']}")
                st.markdown(message["content"])

def handle_user_message(user_message):
    """Handle user message and generate AI response"""
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Generate AI response
    with st.spinner("🤔 AI Doctor is thinking..."):
        try:
            # Create context-aware query
            context_query = f"""
            User Profile: {st.session_state.user_profile}
            Medical Context: {st.session_state.medical_context[:1000]}...
            
            User Question: {user_message}
            
            Please provide a helpful, accurate, and personalized medical response based on the user's profile and uploaded medical documents. 
            Always include appropriate medical disclaimers and recommend consulting healthcare professionals when necessary.
            """
            
            # Get AI response
            result = st.session_state.doc_crew.crew().kickoff(inputs={"user_query": context_query})
            
            # Add AI response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": str(result),
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
    
    st.rerun()

def show_medical_context():
    """Show medical context sidebar"""
    st.markdown("#### 👤 Your Profile")
    
    if st.session_state.user_profile and any(v for v in st.session_state.user_profile.values() if v is not None):
        profile = st.session_state.user_profile
        st.write(f"**Age:** {profile.get('age', 'N/A')}")
        st.write(f"**Gender:** {profile.get('gender', 'N/A')}")
        if profile.get('medical_conditions'):
            st.write(f"**Conditions:** {profile['medical_conditions'][:50]}...")
        if profile.get('current_medications'):
            st.write(f"**Medications:** {profile['current_medications'][:50]}...")
    else:
        st.info("No profile information provided")
    
    st.markdown("#### 📄 Medical Context")
    if st.session_state.chat_mode == "general":
        st.info("General medical chat mode")
    elif st.session_state.processed_documents:
        st.write(f"**Documents:** {len(st.session_state.processed_documents)} uploaded")
        for i, doc in enumerate(st.session_state.processed_documents):
            with st.expander(f"📄 {doc['filename']}", expanded=False):
                st.write(f"**Size:** {doc['size_kb']:.1f} KB")
                st.write(f"**Processed:** {doc['processed_at']}")
                st.write(f"**Summary:** {doc['content_summary'][:100]}...")
    else:
        st.info("No documents uploaded yet")

def upload_additional_documents():
    """Handle additional document uploads"""
    st.markdown("#### 📄 Upload Additional Documents")
    
    new_files = st.file_uploader(
        "Choose additional medical files",
        type=['pdf'],
        accept_multiple_files=True,
        key="additional_files"
    )
    
    if new_files and st.button("Process Additional Files"):
        process_documents_for_chat(new_files)

def help_info_tab():
    """Help and Information Tab"""
    st.header("ℹ️ Help & Information")
    
    st.markdown("""
    ## 🏥 AI Doctor Crew - User Guide
    
    ### 📄 Document Processing
    1. **Upload PDFs**: Upload medical documents, research papers, clinical guidelines
    2. **Process**: Click "Process Documents" to extract and embed content
    3. **Monitor**: Track processing status and view processed documents
    
    ### 🔍 Medical Analysis
    Choose from five analysis types:
    
    #### 🩺 General Medical Query
    - Enter any medical question or query
    - Specify patient demographics and urgency
    - Get comprehensive medical analysis
    
    #### 🩺 Symptom Analysis  
    - Describe patient symptoms in detail
    - Provide symptom characteristics (duration, severity, onset)
    - Receive differential diagnoses and recommendations
    
    #### 💊 Drug Interaction Check
    - List current medications
    - Check for drug-drug, drug-food, drug-condition interactions
    - Get safety recommendations
    
    #### 📋 Treatment Protocol Research
    - Research evidence-based treatment protocols
    - Filter by guidelines, evidence level, population
    - Get current best practices
    
    #### 📚 Medical Literature Review
    - Comprehensive literature review on medical topics
    - Filter by study types and time periods
    - Get evidence synthesis
    
    ### 📊 Analysis History
    - View all completed analyses
    - Filter by type and time period
    - Export history as CSV
    
    ### ⚙️ System Settings
    - Check environment configuration
    - Monitor system status
    - Clear data and restart system
    
    ### ⚠️ Important Notes
    - This system is for educational and research purposes only
    - Always consult healthcare professionals for medical decisions
    - Ensure proper environment variable setup
    - Keep your API keys secure
    
    ### 🔧 Technical Requirements
    - Python 3.11+
    - Google API key for Gemini embeddings
    - Milvus vector database access
    - Required Python packages (see requirements.txt)
    
    ### 🆘 Troubleshooting
    1. **Initialization Errors**: Check environment variables
    2. **Processing Failures**: Verify PDF file integrity
    3. **Analysis Errors**: Check internet connectivity and API limits
    4. **Performance Issues**: Consider document size and complexity
    
    ### 📞 Support
    For technical support or questions:
    - Check the system status in the sidebar
    - Review error messages carefully  
    - Ensure all dependencies are installed
    - Verify environment configuration
    """)

def export_chat_history():
    """Export chat history functionality"""
    if not st.session_state.chat_history:
        st.info("No chat history to export")
        return
    
    import json
    
    # Create export data
    export_data = {
        "export_date": datetime.now().isoformat(),
        "user_profile": st.session_state.user_profile,
        "chat_history": st.session_state.chat_history,
        "documents_processed": len(st.session_state.processed_documents)
    }
    
    # Convert to JSON
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="📄 Download Chat History",
        data=json_str,
        file_name=f"ai_doctor_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
