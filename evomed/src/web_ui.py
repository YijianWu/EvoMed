"""
医疗诊断系统 Web UI
支持简化版（diagnosis_api）和全量版（main_diagnosis_pipeline）
"""

import streamlit as st
import json
import sys
import os
from typing import Optional, Dict, Any
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 简化版 API
from diagnosis_api import DiagnosisAPI, convert_patient_json_to_text

# 全量版 API（需要时导入）
try:
    from main_diagnosis_pipeline import DiagnosticPipeline, PatientInfo
    FULL_VERSION_AVAILABLE = True
except ImportError as e:
    st.warning(f"全量版本不可用: {e}")
    FULL_VERSION_AVAILABLE = False


def create_example_data():
    """创建示例数据"""
    patient_example = {
        "patientName": "温惠远",
        "patientGender": "男",
        "patientAge": 67,
        "chiefComplaint": "肚子疼五小时前突然出现",
        "presentIllness": "患者为67岁男性，已婚。五小时前开始出现右中下腹疼痛，并逐渐扩散至左侧腹部，疼痛性质为钝痛且影响日常活动及饮食睡眠。患者近期无性生活史，无孕产史。目前主要症状为肚子疼和恶心呕吐。患者否认以往有类似肚子疼的经历，否认消化不良、肠胃炎等病史，也无慢性疾病或重大手术史。",
        "personalHistory": "否认消化系统疾病、慢性疾病或其他相关病史。",
        "labs": [
            {"key": "血常规", "value": "WBC 11.2, N% 84.0"}
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
    """解析 JSON 输入"""
    if not json_str or json_str.strip() == "":
        return None
    
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        st.error(f"{field_name} 格式错误: {e}")
        return None


def run_simplified_diagnosis(patient_data: Dict, c_data: Optional[Dict], ecg_data: Optional[Dict]) -> tuple:
    """运行简化版诊断"""
    try:
        # 初始化 API
        with st.spinner("正在初始化诊断服务..."):
            api = DiagnosisAPI()
        
        # 转换患者数据
        patient_text = convert_patient_json_to_text(patient_data)
        
        # 执行诊断
        with st.spinner("专家诊断中，请稍候..."):
            output = api.diagnose(patient_text, c_data, ecg_data, max_experts=5)
        
        # 生成两种输出格式
        diagnosis_json = output.to_doctor_response()
        doctor_json = output.to_management_response(
            clinic_code=patient_data.get('clinicCode', '')
        )
        
        return diagnosis_json, doctor_json, None
        
    except Exception as e:
        st.error(f"诊断失败: {e}")
        import traceback
        return None, None, traceback.format_exc()


def run_full_diagnosis(patient_data: Dict, c_data: Optional[Dict], ecg_data: Optional[Dict],
                      enable_rag: bool, enable_experience: bool, enable_case: bool) -> tuple:
    """运行全量版诊断"""
    try:
        # 初始化流水线
        with st.spinner("正在初始化全量诊断流水线（含 RAG/经验库/病例库）..."):
            pipeline = DiagnosticPipeline(
                activation_mode="evolved_pool",
                evolved_pool_path="outputs/moa_optimized_expert_pool_64.json",
                enable_rag=enable_rag,
                enable_experience=enable_experience,
                enable_case=enable_case
            )
        
        # 构建 PatientInfo 对象
        patient_info = PatientInfo(
            patient_id=patient_data.get('patientName', 'UNKNOWN'),
            gender=patient_data.get('patientGender', '未知'),
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
        
        # 执行诊断
        with st.spinner("多专家会诊中，请稍候..."):
            results = pipeline.run_pipeline(patient_info, top_k=8)
        
        # 提取关键信息
        step4_output = results.get('steps', {}).get('step4', {}).get('output', '')
        expert_opinions = results.get('steps', {}).get('expert_opinions', [])
        
        # 简化输出格式（类似 diagnosis_api 的输出）
        diagnosis_json = {
            "status": "success",
            "patient_info": {
                "diagnosis_result": step4_output[:1000] if step4_output else "诊断输出为空"
            },
            "expert_opinions": expert_opinions,
            "full_results": results
        }
        
        doctor_json = {
            "type": "DiagnosticMessage",
            "summary_value": f"全量版诊断完成。激活 {len(expert_opinions)} 位专家。",
            "clinic_code": patient_data.get('clinicCode', ''),
            "diagnostic_result_value": step4_output[:500] if step4_output else "",
            "full_results": results
        }
        
        return diagnosis_json, doctor_json, None
        
    except Exception as e:
        st.error(f"全量诊断失败: {e}")
        import traceback
        return None, None, traceback.format_exc()


def main():
    st.set_page_config(
        page_title="医疗诊断系统 Web UI",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 多专科医疗诊断系统 Web UI")
    st.markdown("---")
    
    # 侧边栏 - 配置选项
    with st.sidebar:
        st.header("⚙️ 配置选项")
        
        # 选择版本
        use_full_version = st.checkbox(
            "使用全量版本（包含 RAG/经验库/病例库）",
            value=False,
            help="简化版速度快但功能有限，全量版功能完整但速度较慢"
        )
        
        if use_full_version and FULL_VERSION_AVAILABLE:
            st.subheader("知识检索配置")
            enable_rag = st.checkbox("启用 RAG 指南检索", value=True)
            enable_experience = st.checkbox("启用经验库检索", value=True)
            enable_case = st.checkbox("启用病例库检索", value=True)
        else:
            enable_rag = enable_experience = enable_case = False
        
        st.markdown("---")
        st.subheader("📖 使用说明")
        st.markdown("""
1. 在下方输入框中填入 JSON 格式的数据
2. 或点击"加载示例数据"快速测试
3. 点击"开始诊断"进行分析
4. 查看诊断结果并下载 JSON 文件
        """)
    
    # 主界面
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 输入数据")
        
        # 加载示例按钮
        if st.button("加载示例数据", type="secondary"):
            patient_ex, c_ex, ecg_ex = create_example_data()
            st.session_state['patient_json'] = json.dumps(patient_ex, ensure_ascii=False, indent=2)
            st.session_state['c_json'] = json.dumps(c_ex, ensure_ascii=False, indent=2)
            st.session_state['ecg_json'] = json.dumps(ecg_ex, ensure_ascii=False, indent=2)
            st.rerun()
        
        # 患者信息输入
        st.markdown("##### 1️⃣ 患者信息 (patient.json)")
        patient_json = st.text_area(
            "患者病历数据",
            value=st.session_state.get('patient_json', ''),
            height=200,
            placeholder='{"patientName": "张三", "patientGender": "男", ...}',
            key="patient_input"
        )
        
        # 肠鸣音输入
        st.markdown("##### 2️⃣ 肠鸣音检测 (c.json) - 可选")
        c_json = st.text_area(
            "肠鸣音预测结果",
            value=st.session_state.get('c_json', ''),
            height=120,
            placeholder='{"fold": "fold_0", "pid": "TEST_001", "pred": 0, ...}',
            key="c_input"
        )
        
        # ECG输入
        st.markdown("##### 3️⃣ ECG检测 (ecg.json) - 可选")
        ecg_json = st.text_area(
            "ECG预测结果",
            value=st.session_state.get('ecg_json', ''),
            height=120,
            placeholder='{"pid": "TEST_001", "pred": false, "conf": 0.85, ...}',
            key="ecg_input"
        )
        
        # 诊断按钮
        st.markdown("---")
        diagnose_button = st.button("🚀 开始诊断", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 诊断结果")
        
        if diagnose_button:
            # 解析输入
            patient_data = parse_json_input(patient_json, "患者信息")
            c_data = parse_json_input(c_json, "肠鸣音数据") if c_json.strip() else None
            ecg_data = parse_json_input(ecg_json, "ECG数据") if ecg_json.strip() else None
            
            if not patient_data:
                st.error("❌ 患者信息不能为空！")
            else:
                # 执行诊断
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
                    st.error("诊断过程中出现错误：")
                    st.code(error)
                elif diagnosis_json and doctor_json:
                    st.success("✅ 诊断完成！")
                    
                    # 显示结果选项卡
                    tab1, tab2 = st.tabs(["📋 Diagnosis.json", "👨‍⚕️ Doctor.json"])
                    
                    with tab1:
                        st.json(diagnosis_json)
                        st.download_button(
                            label="📥 下载 diagnosis.json",
                            data=json.dumps(diagnosis_json, ensure_ascii=False, indent=2),
                            file_name="diagnosis.json",
                            mime="application/json"
                        )
                    
                    with tab2:
                        st.json(doctor_json)
                        st.download_button(
                            label="📥 下载 doctor.json",
                            data=json.dumps(doctor_json, ensure_ascii=False, indent=2),
                            file_name="doctor.json",
                            mime="application/json"
                        )
                    
                    # 存储结果到 session_state
                    st.session_state['last_diagnosis'] = diagnosis_json
                    st.session_state['last_doctor'] = doctor_json
        
        # 如果之前有诊断结果，继续显示
        elif 'last_diagnosis' in st.session_state:
            st.info("显示上一次的诊断结果")
            
            tab1, tab2 = st.tabs(["📋 Diagnosis.json", "👨‍⚕️ Doctor.json"])
            
            with tab1:
                st.json(st.session_state['last_diagnosis'])
                st.download_button(
                    label="📥 下载 diagnosis.json",
                    data=json.dumps(st.session_state['last_diagnosis'], ensure_ascii=False, indent=2),
                    file_name="diagnosis.json",
                    mime="application/json"
                )
            
            with tab2:
                st.json(st.session_state['last_doctor'])
                st.download_button(
                    label="📥 下载 doctor.json",
                    data=json.dumps(st.session_state['last_doctor'], ensure_ascii=False, indent=2),
                    file_name="doctor.json",
                    mime="application/json"
                )
        else:
            st.info("👆 请在左侧输入数据并点击 '开始诊断'")
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>
        ⚠️ 免责声明：本系统仅用于医疗诊断辅助和学术研究，不能替代专业医生的诊断和治疗。<br>
        所有诊断结果仅供参考，请以专业医疗机构的诊断为准。
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

