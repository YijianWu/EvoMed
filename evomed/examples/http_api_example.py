"""
HTTP API 使用示例

演示如何通过 HTTP API 进行医疗诊断

使用前请确保 API 服务器已启动:
    python src/api_server.py
"""

import requests
import json


# API 服务器地址
API_BASE_URL = "http://localhost:8000"


def check_health():
    """检查服务器健康状态"""
    print("="*60)
    print("1. 健康检查")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    result = response.json()
    
    print(f"状态: {result['status']}")
    print(f"消息: {result['message']}")
    print(f"专家数量: {result.get('expert_count', 'N/A')}")
    print()


def list_experts():
    """获取专家池信息"""
    print("="*60)
    print("2. 查询专家池")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/api/v1/experts")
    result = response.json()
    
    print(f"总专家数: {result['total_experts']}")
    print(f"总专科数: {result['total_specialties']}")
    print("\n各专科专家:")
    
    for specialty, experts in result['by_specialty'].items():
        print(f"\n  {specialty}:")
        for expert in experts:
            print(f"    - {expert['name']} (fitness={expert['fitness']:.3f})")
    print()


def list_specialties():
    """获取专科列表"""
    print("="*60)
    print("3. 查询专科列表")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/api/v1/specialties")
    result = response.json()
    
    print(f"总专科数: {result['total']}")
    print("\n专科详情:")
    
    for specialty, info in result['specialties'].items():
        print(f"  {specialty}: {info['expert_count']}位专家, "
              f"平均fitness={info['avg_fitness']:.3f}")
    print()


def diagnose_patient_basic():
    """基础诊断示例（患者端）"""
    print("="*60)
    print("4. 基础诊断示例（患者端）")
    print("="*60)
    
    # 构建患者数据
    patient_data = {
        "patient": {
            "patientName": "张三",
            "patientGender": "男",
            "patientAge": 52,
            "chiefComplaint": "腹痛伴恶心呕吐2天",
            "presentIllness": """患者2天前无明显诱因出现上腹部疼痛，呈阵发性绞痛，
向右肩背部放射，伴恶心呕吐，呕吐物为胃内容物，非喷射性。
疼痛与进食油腻食物有关。无发热，无黄疸，大小便正常。""",
            "personalHistory": "高血压病史5年，规律服用降压药，血压控制良好。",
            "physical_examination": "T: 37.2℃, P: 88次/分, R: 18次/分, BP: 135/85mmHg, 右上腹压痛（+），Murphy征阳性",
            "labs": [
                {
                    "key": "白细胞计数",
                    "value": "11.2×10^9/L（偏高）"
                },
                {
                    "key": "总胆红素",
                    "value": "22 μmol/L（略高）"
                }
            ]
        },
        "max_experts": 5
    }
    
    print("发送诊断请求...")
    response = requests.post(
        f"{API_BASE_URL}/api/v1/diagnose/patient",
        json=patient_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ 诊断完成！\n")
        
        # 打印患者可见信息
        patient_info = result.get('patient_info', {})
        
        print("【诊断Top5】")
        print(patient_info.get('diagnosis_top5', ''))
        
        print("\n【诊断依据】")
        print(patient_info.get('diagnosis_basis', ''))
        
        print("\n【鉴别诊断】")
        print(patient_info.get('differential_diagnosis', ''))
        
        print("\n【建议的检查或检验】")
        print(patient_info.get('suggested_examinations', ''))
        
    else:
        print(f"❌ 诊断失败: {response.status_code}")
        print(response.text)
    
    print()


def diagnose_doctor_full():
    """完整诊断示例（医生端，含风险评估）"""
    print("="*60)
    print("5. 完整诊断示例（医生端 + 风险评估）")
    print("="*60)
    
    # 构建完整数据（包含肠鸣音和ECG）
    patient_data = {
        "patient": {
            "patientName": "李四",
            "patientGender": "女",
            "patientAge": 45,
            "chiefComplaint": "右上腹痛3天",
            "presentIllness": """患者3天前无明显诱因出现右上腹痛，呈持续性胀痛，
向右肩背部放射，伴恶心、呕吐，进食油腻食物后症状加重。无发热、黄疸。""",
            "personalHistory": "既往体健，否认高血压、糖尿病史",
            "physical_examination": """体温37.2℃，血压120/80mmHg，心率80次/分
右上腹压痛阳性，墨菲氏征阳性，肝脾未触及""",
            "labs": [
                {
                    "key": "白细胞计数",
                    "value": "12.5×10^9/L（偏高）"
                },
                {
                    "key": "中性粒细胞",
                    "value": "82%"
                }
            ],
            "clinicCode": "CLINIC_001"
        },
        "bowel_sound": {
            "pred": 0,
            "prob_0": 0.72,
            "prob_1": 0.28
        },
        "ecg": {
            "pred": False,
            "conf": 0.85
        },
        "max_experts": 5
    }
    
    print("发送诊断请求（含多模态数据）...")
    response = requests.post(
        f"{API_BASE_URL}/api/v1/diagnose/doctor",
        json=patient_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ 诊断完成！\n")
        
        # 打印风险评估（医生专属）
        if 'risk_assessment' in result:
            risk = result['risk_assessment']
            print("【危险分层】（医生专属）")
            print(f"风险等级: {risk.get('risk_level', '未评估')}")
            print(f"居家观测: {risk.get('home_monitoring', '')}")
            print(f"就诊意见: {risk.get('visit_advice', '')}")
            print(f"是否需要急救: {risk.get('emergency_needed', '')}")
        
        # 打印医生端管理格式
        if 'diagnostic_result_value' in result:
            print("\n【管理端格式】")
            print(f"总结: {result.get('summary_value', '')}")
            print(f"\n诊断结果:\n{result.get('diagnostic_result_value', '')}")
            print(f"\n病情分析:\n{result.get('condition_analysis_value', '')}")
            print(f"\n检查建议:\n{result.get('suggestions_examinations_value', '')}")
        
    else:
        print(f"❌ 诊断失败: {response.status_code}")
        print(response.text)
    
    print()


def save_result_to_file():
    """保存诊断结果到文件"""
    print("="*60)
    print("6. 保存诊断结果到文件")
    print("="*60)
    
    patient_data = {
        "patient": {
            "patientGender": "男",
            "patientAge": 52,
            "chiefComplaint": "腹痛伴恶心呕吐2天",
            "presentIllness": "患者2天前无明显诱因出现上腹部疼痛，呈阵发性绞痛"
        },
        "max_experts": 3
    }
    
    print("发送诊断请求...")
    response = requests.post(
        f"{API_BASE_URL}/api/v1/diagnose/doctor",
        json=patient_data
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # 保存到文件
        output_file = "diagnosis_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 诊断结果已保存到: {output_file}")
    else:
        print(f"❌ 诊断失败: {response.status_code}")
    
    print()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("HTTP API 使用示例")
    print("="*60)
    print(f"API 服务器地址: {API_BASE_URL}")
    print("="*60 + "\n")
    
    try:
        # 1. 健康检查
        check_health()
        
        # 2. 查询专家池
        list_experts()
        
        # 3. 查询专科
        list_specialties()
        
        # 4. 基础诊断（患者端）
        diagnose_patient_basic()
        
        # 5. 完整诊断（医生端）
        diagnose_doctor_full()
        
        # 6. 保存结果到文件
        save_result_to_file()
        
        print("="*60)
        print("✅ 所有示例运行完成！")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到 API 服务器")
        print(f"请确保服务器已启动: python src/api_server.py")
        print(f"服务器地址: {API_BASE_URL}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


