"""
HTTP API Server
HTTP interface for medical diagnosis service based on FastAPI

Usage:
    python src/api_server.py --host 0.0.0.0 --port 8000

Access API Documentation:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import argparse
import json
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evomed.pipeline import DiagnosisAPI, diagnose_patient


# ============================================================
# API Data Models
# ============================================================

class PatientData(BaseModel):
    """Patient medical record data"""
    patientName: Optional[str] = Field(None, description="Patient Name")
    patientGender: Optional[str] = Field(None, description="Patient Gender (F/M)")
    patientAge: Optional[int] = Field(None, description="Patient Age")
    chiefComplaint: Optional[str] = Field(None, description="Chief Complaint")
    presentIllness: Optional[str] = Field(None, description="History of Present Illness")
    personalHistory: Optional[str] = Field(None, description="Past History")
    physical_examination: Optional[Any] = Field(None, description="Physical Examination")
    labs: Optional[Any] = Field(None, description="Laboratory indicators")
    exam: Optional[Any] = Field(None, description="Examination indicators")
    
    # Compatible with table input format
    gender: Optional[str] = Field(None, description="Gender (F/M)")
    age: Optional[int] = Field(None, description="Age")
    history: Optional[Dict] = Field(None, description="History information")
    
    # Contextual information
    clinicCode: Optional[str] = Field(None, description="Clinic Code")


class BowelSoundData(BaseModel):
    """Bowel sound prediction results"""
    fold: Optional[str] = ""
    pid: Optional[str] = ""
    pred: Optional[int] = Field(0, description="Prediction result: 0=Normal, 1=Abnormal")
    prob_0: Optional[float] = Field(0.5, description="Probability of normal")
    prob_1: Optional[float] = Field(0.5, description="Probability of abnormal")


class ECGData(BaseModel):
    """ECG prediction results"""
    pid: Optional[str] = ""
    path: Optional[str] = ""
    pred_id: Optional[int] = 0
    pred: Optional[bool] = Field(False, description="Prediction result: False=Normal, True=Abnormal")
    conf: Optional[float] = Field(0.5, description="Confidence")
    topk: Optional[List] = []
    probs_json: Optional[str] = ""


class DiagnosisRequest(BaseModel):
    """Diagnosis Request"""
    patient: PatientData = Field(..., description="Patient medical record data")
    bowel_sound: Optional[BowelSoundData] = Field(None, description="Bowel sound detection results (optional)")
    ecg: Optional[ECGData] = Field(None, description="ECG detection results (optional)")
    max_experts: Optional[int] = Field(5, description="Maximum number of activated experts", ge=1, le=10)
    for_doctor: Optional[bool] = Field(False, description="Whether to return in doctor's format")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    expert_count: Optional[int] = None


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Multi-Specialty Medical Diagnosis Service API",
    description="Medical diagnosis auxiliary system based on Evolving Expert Pool (EEP)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Please modify to specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global DiagnosisAPI instance (initialized at startup)
diagnosis_api: Optional[DiagnosisAPI] = None


@app.on_event("startup")
async def startup_event():
    """Initialize at service startup"""
    global diagnosis_api
    print("="*60)
    print("Starting Medical Diagnosis Service...")
    print("="*60)
    try:
        diagnosis_api = DiagnosisAPI()
        print("✅ Diagnosis service initialized successfully")
    except Exception as e:
        print(f"❌ Diagnosis service initialization failed: {e}")
        raise


@app.get("/", tags=["Health Check"])
async def root():
    """Root path"""
    return {
        "message": "Multi-Specialty Medical Diagnosis Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    """Health check interface"""
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="Diagnosis service not initialized")
    
    expert_count = len(diagnosis_api.expert_pool_28) if diagnosis_api.expert_pool_28 else 0
    
    return HealthResponse(
        status="healthy",
        message="Service running normally",
        expert_count=expert_count
    )


@app.post("/api/v1/diagnose", tags=["Diagnosis Service"])
async def diagnose(request: DiagnosisRequest):
    """
    Execute medical diagnosis
    
    Input:
    - patient: Patient medical record data (required)
    - bowel_sound: Bowel sound detection results (optional)
    - ecg: ECG detection results (optional)
    - max_experts: Maximum number of activated experts (optional, default 5)
    - for_doctor: Whether to return in doctor's format (optional, default False)
    
    Output:
    - Patient-side format: Includes diagnosis results, diagnosis basis, differential diagnosis, recommended examinations
    - Doctor-side format: Additionally includes risk assessment, danger stratification, visit suggestions, etc.
    """
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="Diagnosis service not initialized")
    
    try:
        # Convert request data
        patient_dict = request.patient.dict(exclude_none=True)
        
        bowel_sound_dict = None
        if request.bowel_sound:
            bowel_sound_dict = request.bowel_sound.dict(exclude_none=True)
        
        ecg_dict = None
        if request.ecg:
            ecg_dict = request.ecg.dict(exclude_none=True)
        
        # Execute diagnosis
        result = diagnose_patient(
            patient_record=patient_dict,
            bowel_sound_result=bowel_sound_dict,
            ecg_result=ecg_dict,
            for_doctor=request.for_doctor,
            max_experts=request.max_experts
        )
        
        return result
        
    except Exception as e:
        print(f"Error during diagnosis process: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")


@app.post("/api/v1/diagnose/patient", tags=["Diagnosis Service"])
async def diagnose_patient_view(request: DiagnosisRequest):
    """
    Patient-side diagnosis interface (returns patient-visible information)
    """
    request.for_doctor = False
    return await diagnose(request)


@app.post("/api/v1/diagnose/doctor", tags=["Diagnosis Service"])
async def diagnose_doctor_view(request: DiagnosisRequest):
    """
    Doctor-side diagnosis interface (returns full information, including risk assessment)
    """
    request.for_doctor = True
    return await diagnose(request)


@app.get("/api/v1/experts", tags=["Expert Pool Management"])
async def list_experts():
    """Get expert pool information"""
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="Diagnosis service not initialized")
    
    experts = []
    for expert in diagnosis_api.expert_pool_28:
        experts.append({
            "id": expert.get("id"),
            "name": expert.get("name"),
            "specialty": expert.get("specialty"),
            "description": expert.get("description"),
            "fitness": expert.get("fitness", 0),
            "accuracy": expert.get("accuracy", 0),
            "avg_score": expert.get("avg_score", 0)
        })
    
    # Group by specialty
    specialties = {}
    for expert in experts:
        specialty = expert["specialty"]
        if specialty not in specialties:
            specialties[specialty] = []
        specialties[specialty].append(expert)
    
    return {
        "total_experts": len(experts),
        "total_specialties": len(specialties),
        "experts": experts,
        "by_specialty": specialties
    }


@app.get("/api/v1/specialties", tags=["Expert Pool Management"])
async def list_specialties():
    """Get list of all specialties"""
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="Diagnosis service not initialized")
    
    specialties = {}
    for specialty, experts in diagnosis_api.specialty_to_experts.items():
        specialties[specialty] = {
            "expert_count": len(experts),
            "avg_fitness": sum(e.get("fitness", 0) for e in experts) / len(experts) if experts else 0
        }
    
    return {
        "total": len(specialties),
        "specialties": specialties
    }


# ============================================================
# Start Service
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Specialty Medical Diagnosis Service HTTP API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (development mode)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Starting Medical Diagnosis Service HTTP API")
    print(f"Address: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"ReDoc Docs: http://{args.host}:{args.port}/redoc")
    print("="*60)
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
