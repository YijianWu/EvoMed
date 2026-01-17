"""
HTTP API 服务器
基于 FastAPI 实现的医疗诊断服务 HTTP 接口

使用方法:
    python src/api_server.py --host 0.0.0.0 --port 8000

访问 API 文档:
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diagnosis_api import DiagnosisAPI, diagnose_patient


# ============================================================
# API 数据模型
# ============================================================

class PatientData(BaseModel):
    """患者病历数据"""
    patientName: Optional[str] = Field(None, description="患者姓名")
    patientGender: Optional[str] = Field(None, description="患者性别 (F/M/女/男)")
    patientAge: Optional[int] = Field(None, description="患者年龄")
    chiefComplaint: Optional[str] = Field(None, description="主诉")
    presentIllness: Optional[str] = Field(None, description="现病史")
    personalHistory: Optional[str] = Field(None, description="既往史")
    physical_examination: Optional[Any] = Field(None, description="体格检查")
    labs: Optional[Any] = Field(None, description="检验指标")
    exam: Optional[Any] = Field(None, description="检查指标")
    
    # 兼容表格输入格式
    gender: Optional[str] = Field(None, description="性别 (F/M)")
    age: Optional[int] = Field(None, description="年龄")
    history: Optional[Dict] = Field(None, description="病史信息")
    
    # 上下文信息
    clinicCode: Optional[str] = Field(None, description="诊所代码")


class BowelSoundData(BaseModel):
    """肠鸣音预测结果"""
    fold: Optional[str] = ""
    pid: Optional[str] = ""
    pred: Optional[int] = Field(0, description="预测结果: 0=正常, 1=异常")
    prob_0: Optional[float] = Field(0.5, description="正常概率")
    prob_1: Optional[float] = Field(0.5, description="异常概率")


class ECGData(BaseModel):
    """ECG预测结果"""
    fold: Optional[str] = ""
    pid: Optional[str] = ""
    pred: Optional[int] = Field(0, description="预测结果: 0=正常, 1=异常")
    prob_0: Optional[float] = Field(0.5, description="正常概率")
    prob_1: Optional[float] = Field(0.5, description="异常概率")


class DiagnosisRequest(BaseModel):
    """诊断请求"""
    patient: PatientData = Field(..., description="患者病历数据")
    bowel_sound: Optional[BowelSoundData] = Field(None, description="肠鸣音检测结果（可选）")
    ecg: Optional[ECGData] = Field(None, description="ECG检测结果（可选）")
    max_experts: Optional[int] = Field(5, description="最大激活专家数量", ge=1, le=10)
    for_doctor: Optional[bool] = Field(False, description="是否返回医生端格式")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    message: str
    expert_count: Optional[int] = None


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="多专科医疗诊断服务 API",
    description="基于可演化专家池(EEP)的医疗诊断辅助系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请修改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局诊断API实例（在启动时初始化）
diagnosis_api: Optional[DiagnosisAPI] = None


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化"""
    global diagnosis_api
    print("="*60)
    print("正在启动医疗诊断服务...")
    print("="*60)
    try:
        diagnosis_api = DiagnosisAPI()
        print("✅ 诊断服务初始化成功")
    except Exception as e:
        print(f"❌ 诊断服务初始化失败: {e}")
        raise


@app.get("/", tags=["健康检查"])
async def root():
    """根路径"""
    return {
        "message": "多专科医疗诊断服务 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="诊断服务未初始化")
    
    expert_count = len(diagnosis_api.expert_pool_28) if diagnosis_api.expert_pool_28 else 0
    
    return HealthResponse(
        status="healthy",
        message="服务运行正常",
        expert_count=expert_count
    )


@app.post("/api/v1/diagnose", tags=["诊断服务"])
async def diagnose(request: DiagnosisRequest):
    """
    执行医疗诊断
    
    输入：
    - patient: 患者病历数据（必需）
    - bowel_sound: 肠鸣音检测结果（可选）
    - ecg: ECG检测结果（可选）
    - max_experts: 最大激活专家数量（可选，默认5）
    - for_doctor: 是否返回医生端格式（可选，默认False）
    
    输出：
    - 患者端格式：包含诊断结果、诊断依据、鉴别诊断、建议检查
    - 医生端格式：额外包含风险评估、危险分层、就诊建议等
    """
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="诊断服务未初始化")
    
    try:
        # 转换请求数据
        patient_dict = request.patient.dict(exclude_none=True)
        
        bowel_sound_dict = None
        if request.bowel_sound:
            bowel_sound_dict = request.bowel_sound.dict(exclude_none=True)
        
        ecg_dict = None
        if request.ecg:
            ecg_dict = request.ecg.dict(exclude_none=True)
        
        # 执行诊断
        result = diagnose_patient(
            patient_record=patient_dict,
            bowel_sound_result=bowel_sound_dict,
            ecg_result=ecg_dict,
            for_doctor=request.for_doctor,
            max_experts=request.max_experts
        )
        
        return result
        
    except Exception as e:
        print(f"诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"诊断失败: {str(e)}")


@app.post("/api/v1/diagnose/patient", tags=["诊断服务"])
async def diagnose_patient_view(request: DiagnosisRequest):
    """
    患者端诊断接口（返回患者可见信息）
    """
    request.for_doctor = False
    return await diagnose(request)


@app.post("/api/v1/diagnose/doctor", tags=["诊断服务"])
async def diagnose_doctor_view(request: DiagnosisRequest):
    """
    医生端诊断接口（返回完整信息，包括风险评估）
    """
    request.for_doctor = True
    return await diagnose(request)


@app.get("/api/v1/experts", tags=["专家池管理"])
async def list_experts():
    """获取专家池信息"""
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="诊断服务未初始化")
    
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
    
    # 按专科分组
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


@app.get("/api/v1/specialties", tags=["专家池管理"])
async def list_specialties():
    """获取所有专科列表"""
    if diagnosis_api is None:
        raise HTTPException(status_code=503, detail="诊断服务未初始化")
    
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
# 启动服务
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多专科医疗诊断服务 HTTP API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载（开发模式）")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"启动医疗诊断服务 HTTP API")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print(f"ReDoc 文档: http://{args.host}:{args.port}/redoc")
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


