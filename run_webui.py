"""
Medical Diagnosis System Web UI
Supports both Simplified version (diagnosis_api) and Full version (main_diagnosis_pipeline)
"""

import streamlit as st
import json
import sys
import os
from typing import Optional, Dict, Any
import pandas as pd

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simplified API
from evomed.pipeline import DiagnosisAPI, convert_patient_json_to_text

# Full Version API (import when needed)
try:
    from evomed.diagnosis import DiagnosticPipeline, PatientInfo
    FULL_VERSION_AVAILABLE = True
except ImportError as e:
    st.warning(f"Full version unavailable: {e}")
    FULL_VERSION_AVAILABLE = False


def create_example_data():
    """Create example data"""
    patient_example = {
        "patientName": "Huiyuan Wen",
        "patientGender": "Male",
        "patientAge": 67,
        "chiefComplaint": "Abdominal pain appeared suddenly five hours ago",
        "presentIllness": "The patient is a 67-year-old married male. He started experiencing pain in the right middle and lower abdomen five hours ago, which gradually spread to the left abdomen. The pain is dull and affects daily activities, diet, and sleep. The patient has no recent sexual history and no obstetric history. Current main symptoms are abdominal pain, nausea, and vomiting. The patient denies any previous similar experiences of abdominal pain, denies history of indigestion, gastroenteritis, etc., and has no chronic diseases or major surgical history.",
        "personalHistory": "Denies gastrointestinal diseases, chronic diseases, or other relevant history.",
        "labs": [
            {"key": "Blood Routine", "value": "WBC 11.2, N% 84.0"}
        ],
        "clinicCode": "pre-jqxci99"
    }
    
    c_example = {
        "fold": "fold_0",
        "pid": "TEST_001",
        "pred": 0,
        "prob_0": 0.72,
        "prob_1": 0.28
    }
    
    ecg_example = {
        "pid": "TEST_001",
        "path": "path/to/ecg.jpg",
        "pred": False,
        "conf": 0.85,
        "topk": [["False", 0.85], ["True", 0.15]]
    }
    
    return patient_example, c_example, ecg_example


def parse_json_input(json_str: str, field_name: str) -> Optional[Dict]:
    """Parse JSON input"""
    if not json_str or json_str.strip() == "":
        return None
    
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        st.error(f"{field_name} Format Error: {e}")
        return None


def run_simplified_diagnosis(patient_data: Dict, c_data: Optional[Dict], ecg_data: Optional[Dict]) -> tuple:
    """Run simplified diagnosis"""
    try:
        # Initialize API
        with st.spinner("Initializing diagnosis service..."):
            api = DiagnosisAPI()
        
        # Convert patient data
        patient_text = convert_patient_json_to_text(patient_data)
        
        # Execute diagnosis
        with st.spinner("Expert diagnosis in progress, please wait..."):
            output = api.diagnose(patient_text, c_data, ecg_data, max_experts=5)
        
        # Generate two output formats
        diagnosis_json = output.to_doctor_response()
        doctor_json = output.to_management_response(
            clinic_code=patient_data.get('clinicCode', '')
        )
        
        return diagnosis_json, doctor_json, None
        
    except Exception as e:
        st.error(f"Diagnosis failed: {e}")
        import traceback
        return None, None, traceback.format_exc()


def run_full_diagnosis(patient_data: Dict, c_data: Optional[Dict], ecg_data: Optional[Dict],
                      enable_rag: bool, enable_experience: bool, enable_case: bool) -> tuple:
    """Run full diagnosis"""
    try:
        # Initialize pipeline
        with st.spinner("Initializing full diagnosis pipeline (inc. RAG/Experience/Case Library)..."):
            pipeline = DiagnosticPipeline(
                activation_mode="evolved_pool",
                evolved_pool_path="outputs/moa_optimized_expert_pool_64.json",
                enable_rag=enable_rag,
                enable_experience=enable_experience,
                enable_case=enable_case
            )
        
        # Build PatientInfo object
        patient_info = PatientInfo(
            patient_id=patient_data.get('patientName', 'UNKNOWN'),
            gender=patient_data.get('patientGender', 'Unknown'),
            age=patient_data.get('patientAge', 0),
            department='',
            chief_complaint=patient_data.get('chiefComplaint', ''),
            history_of_present_illness=patient_data.get('presentIllness', ''),
            past_history=patient_data.get('personalHistory', ''),
            personal_history='',
            physical_examination=patient_data.get('physical_examination', ''),
            labs=str(patient_data.get('labs', '')),
            imaging=str(patient_data.get('exam', '')),
            main_diagnosis='',
            main_diagnosis_icd=''
        )
        
        # Execute diagnosis
        with st.spinner("Multi-expert consultation in progress, please wait..."):
            results = pipeline.run_pipeline(patient_info, top_k=8)
        
        # Extract key information
        step4_output = results.get('steps', {}).get('step4', {}).get('output', '')
        expert_opinions = results.get('steps', {}).get('expert_opinions', [])
        
        # Simplify output format (similar to diagnosis_api output)
        diagnosis_json = {
            "status": "success",
            "patient_info": {
                "diagnosis_result": step4_output[:1000] if step4_output else "Diagnosis output is empty"
            },
            "expert_opinions": expert_opinions,
            "full_results": results
        }
        
        doctor_json = {
            "type": "DiagnosticMessage",
            "summary_value": f"Full diagnosis completed. Activated {len(expert_opinions)} experts.",
            "clinic_code": patient_data.get('clinicCode', ''),
            "diagnostic_result_value": step4_output[:500] if step4_output else "",
            "full_results": results
        }
        
        return diagnosis_json, doctor_json, None
        
    except Exception as e:
        st.error(f"Full diagnosis failed: {e}")
        import traceback
        return None, None, traceback.format_exc()


def main():
    st.set_page_config(
        page_title="Medical Diagnosis System Web UI",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Multi-Specialty Medical Diagnosis System Web UI")
    st.markdown("---")
    
    # Sidebar - Configuration Options
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Select version
        use_full_version = st.checkbox(
            "Use Full Version (inc. RAG/Experience/Case)",
            value=False,
            help="Simplified version is faster but limited; Full version is complete but slower."
        )
        
        if use_full_version and FULL_VERSION_AVAILABLE:
            st.subheader("Knowledge Retrieval Config")
            enable_rag = st.checkbox("Enable RAG (Guidelines)", value=True)
            enable_experience = st.checkbox("Enable Experience Library", value=True)
            enable_case = st.checkbox("Enable Case Library", value=True)
        else:
            enable_rag = enable_experience = enable_case = False
        
        st.markdown("---")
        st.subheader("📖 Instructions")
        st.markdown("""
1. Enter data in JSON format below.
2. Or click "Load Example Data" for a quick test.
3. Click "Start Diagnosis" to analyze.
4. View results and download JSON files.
        """)
    
    # Main Interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Data")
        
        # Load Example Button
        if st.button("Load Example Data", type="secondary"):
            patient_ex, c_ex, ecg_ex = create_example_data()
            st.session_state['patient_json'] = json.dumps(patient_ex, ensure_ascii=False, indent=2)
            st.session_state['c_json'] = json.dumps(c_ex, ensure_ascii=False, indent=2)
            st.session_state['ecg_json'] = json.dumps(ecg_ex, ensure_ascii=False, indent=2)
            st.rerun()
        
        # Patient Info Input
        st.markdown("##### 1️⃣ Patient Info (patient.json)")
        patient_json = st.text_area(
            "Patient Medical Record",
            value=st.session_state.get('patient_json', ''),
            height=200,
            placeholder='{"patientName": "John Doe", "patientGender": "Male", ...}',
            key="patient_input"
        )
        
        # Bowel Sounds Input
        st.markdown("##### 2️⃣ Bowel Sounds (c.json) - Optional")
        c_json = st.text_area(
            "Bowel Sounds Prediction",
            value=st.session_state.get('c_json', ''),
            height=120,
            placeholder='{"fold": "fold_0", "pid": "TEST_001", "pred": 0, ...}',
            key="c_input"
        )
        
        # ECG Input
        st.markdown("##### 3️⃣ ECG Detection (ecg.json) - Optional")
        ecg_json = st.text_area(
            "ECG Prediction",
            value=st.session_state.get('ecg_json', ''),
            height=120,
            placeholder='{"pid": "TEST_001", "pred": false, "conf": 0.85, ...}',
            key="ecg_input"
        )
        
        # Diagnosis Button
        st.markdown("---")
        diagnose_button = st.button("🚀 Start Diagnosis", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Diagnostic Results")
        
        if diagnose_button:
            # Parse input
            patient_data = parse_json_input(patient_json, "Patient Info")
            c_data = parse_json_input(c_json, "Bowel Sounds Data") if c_json.strip() else None
            ecg_data = parse_json_input(ecg_json, "ECG Data") if ecg_json.strip() else None
            
            if not patient_data:
                st.error("❌ Patient information cannot be empty!")
            else:
                # Execute diagnosis
                if use_full_version and FULL_VERSION_AVAILABLE:
                    diagnosis_json, doctor_json, error = run_full_diagnosis(
                        patient_data, c_data, ecg_data,
                        enable_rag, enable_experience, enable_case
                    )
                else:
                    diagnosis_json, doctor_json, error = run_simplified_diagnosis(
                        patient_data, c_data, ecg_data
                    )
                
                if error:
                    st.error("Error occurred during diagnosis:")
                    st.code(error)
                elif diagnosis_json and doctor_json:
                    st.success("✅ Diagnosis Complete!")
                    
                    # Display result tabs
                    tab1, tab2 = st.tabs(["📋 Diagnosis.json", "👨‍⚕️ Doctor.json"])
                    
                    with tab1:
                        st.json(diagnosis_json)
                        st.download_button(
                            label="📥 Download diagnosis.json",
                            data=json.dumps(diagnosis_json, ensure_ascii=False, indent=2),
                            file_name="diagnosis.json",
                            mime="application/json"
                        )
                    
                    with tab2:
                        st.json(doctor_json)
                        st.download_button(
                            label="📥 Download doctor.json",
                            data=json.dumps(doctor_json, ensure_ascii=False, indent=2),
                            file_name="doctor.json",
                            mime="application/json"
                        )
                    
                    # Store results in session_state
                    st.session_state['last_diagnosis'] = diagnosis_json
                    st.session_state['last_doctor'] = doctor_json
        
        # If there are previous results, continue showing them
        elif 'last_diagnosis' in st.session_state:
            st.info("Showing last diagnosis result")
            
            tab1, tab2 = st.tabs(["📋 Diagnosis.json", "👨‍⚕️ Doctor.json"])
            
            with tab1:
                st.json(st.session_state['last_diagnosis'])
                st.download_button(
                    label="📥 Download diagnosis.json",
                    data=json.dumps(st.session_state['last_diagnosis'], ensure_ascii=False, indent=2),
                    file_name="diagnosis.json",
                    mime="application/json"
                )
            
            with tab2:
                st.json(st.session_state['last_doctor'])
                st.download_button(
                    label="📥 Download doctor.json",
                    data=json.dumps(st.session_state['last_doctor'], ensure_ascii=False, indent=2),
                    file_name="doctor.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Please input data on the left and click 'Start Diagnosis'")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>
        ⚠️ Disclaimer: This system is for medical diagnosis assistance and academic research only. It cannot replace the diagnosis and treatment of professional doctors.<br>
        All diagnostic results are for reference only. Please refer to the diagnosis of professional medical institutions.
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
