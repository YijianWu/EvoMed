# 多专科医疗诊断系统 - 基于可演化专家池

Multi-Specialty Medical Diagnosis System with Evolvable Expert Pool

> **🎯 推荐使用 Web UI 快速体验！** 
> 
> ```bash
> ./run_webui.sh
> # 访问 http://localhost:8501
> ```
> 
> 📖 详见 [Web UI 快速启动指南](QUICK_START_WEBUI.md)

## 📋 项目概述

本项目提出了一种基于**可演化专家池（Evolvable Expert Pool, EEP）**的多专科医疗诊断辅助系统。通过遗传算法对14个临床科室的专家诊断提示词进行自动优化，构建了包含28位专家的高性能诊断系统。

### 主要特性

- **🧬 可演化专家池**：采用遗传算法自动优化各科室专家的诊断提示词
- **🏥 多专科覆盖**：涵盖14个临床科室，每科室2位专家变体
- **🔄 四步诊断流程**：路由 → 信息重构 → 专家诊断 → 结果聚合
- **📚 知识增强**：集成RAG检索、经验库和病例库
- **🚀 高性能**：最佳专家Fitness达0.976，准确率100%

## 🏗️ 系统架构

```
诊断流程：
输入患者信息
    ↓
Step 1: 科室路由 (system_step1_route.py)
    ↓
Step 2: 信息重构 (system_step2_ir.py)
    ↓
Step 3: 专家诊断 (system_step3_diag.py)
    ↓
Step 4: 结果聚合 (system_step4_agg.py)
    ↓
输出诊断结果
```

## 🏥 14个临床科室

1. **妇产科** - 孕产期管理、妇科肿瘤、月经失调
2. **消化内科** - 胃炎与溃疡、肝脏疾病、功能性胃肠病
3. **儿科** - 呼吸道感染、生长发育评估、小儿消化系统
4. **内分泌科** - 糖尿病管理、甲状腺疾病、骨质疏松
5. **肝胆外科** - 胆石症、肝脏肿瘤、胰腺炎
6. **骨科** - 骨折创伤、关节炎、脊柱疾病
7. **呼吸内科** - 慢性阻塞性肺病、哮喘管理、肺部结节
8. **急诊科** - 生命体征维持、急性中毒、多发伤
9. **泌尿外科** - 泌尿系结石、前列腺疾病、泌尿系肿瘤
10. **全科医学科** - 健康查体解读、常见病初诊、慢病随访
11. **胃肠外科** - 胃肠道肿瘤、阑尾炎、肠梗阻
12. **胸心血管外科** - 肺癌手术、心脏瓣膜病、冠脉搭桥
13. **肿瘤科** - 放化疗方案、肿瘤筛查、癌痛管理
14. **心血管内科** - 高血压、冠心病、心律失常

## 📦 安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/medical-diagnosis-eep.git
cd medical-diagnosis-eep
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境

在 `src/diagnosis_api.py` 中配置您的 API 密钥：

```python
API_BASE_URL = "https://your-api-endpoint.com/v1"
API_KEY = "your-api-key-here"
MODEL_NAME = "gpt-4o"  # 或其他模型
```

## 🚀 快速开始

### 方法 1: Python API 调用

```python
from src.main_diagnosis_pipeline import DiagnosticPipeline, PatientInfo

# 初始化诊断流水线
pipeline = DiagnosticPipeline(
    activation_mode="eep_semantic",  # 使用可演化专家池
    enable_rag=True,                 # 启用RAG检索
    enable_experience=False,
    enable_case=False
)

# 构建患者信息
patient = PatientInfo(
    patient_id="test_001",
    age=45,
    gender="女",
    chief_complaint="腹痛3天",
    present_illness="患者3天前无明显诱因出现腹痛...",
    past_history="既往体健",
    physical_exam="腹部压痛阳性",
    lab_results="WBC 12.5×10^9/L",
    diagnosis=""
)

# 执行诊断
result = pipeline.diagnose(patient)
print(result)
```

### 方法 2: 命令行 (CLI) 使用

#### 基本用法

```bash
# 使用患者数据文件进行诊断
python src/diagnosis_api.py --patient_file patient.json
```

#### 完整参数

```bash
python src/diagnosis_api.py \
    --patient_file patient.json \
    --c_file bowel_sound.json \
    --ecg-file ecg.json
```

#### 患者数据文件格式 (patient.json)

```json
{
  "patientName": "张三",
  "patientGender": "男",
  "patientAge": 52,
  "chiefComplaint": "腹痛伴恶心呕吐2天",
  "presentIllness": "患者2天前无明显诱因出现上腹部疼痛，呈阵发性绞痛...",
  "personalHistory": "高血压病史5年，规律服用降压药",
  "physical_examination": "T: 37.2℃, P: 88次/分, R: 18次/分, BP: 135/85mmHg",
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
}
```

#### 肠鸣音数据格式 (bowel_sound.json)

```json
{
  "fold": "fold_0",
  "pid": "TEST_001",
  "pred": 0,
  "prob_0": 0.72,
  "prob_1": 0.28
}
```

#### ECG数据格式 (ecg.json)

```json
{
  "pid": "TEST_001",
  "path": "path/to/ecg.jpg",
  "pred_id": 0,
  "pred": false,
  "conf": 0.85,
  "topk": [["False", 0.85], ["True", 0.15]]
}
```

#### 输出文件

CLI 会生成两个输出文件：

1. **diagnosis.json** - 完整诊断结果（包含患者信息和风险评估）
2. **doctor.json** - 医生端管理界面格式

### 方法 3: Web UI 界面（推荐）

最简单直观的使用方式，提供可视化的 Web 界面进行诊断。

#### 启动 Web UI

```bash
# 方法 1: 使用启动脚本
./run_webui.sh

# 方法 2: 直接使用 streamlit
streamlit run src/web_ui.py --server.port 8501 --server.address 0.0.0.0
```

#### 访问界面

启动后，在浏览器中访问：
- **本地访问**: http://localhost:8501
- **远程访问**: http://your-server-ip:8501

#### Web UI 功能特性

1. **📝 可视化输入**
   - 提供表单式的 JSON 数据输入
   - 支持患者信息、肠鸣音数据、ECG 数据三项输入
   - 一键加载示例数据快速测试

2. **⚙️ 版本选择**
   - **简化版**：快速诊断，使用 28 专家池
   - **全量版**：完整功能，支持 RAG/经验库/病例库检索
   
3. **📊 实时结果展示**
   - 自动生成 diagnosis.json 和 doctor.json
   - 在线预览 JSON 格式
   - 一键下载诊断结果文件

4. **🎯 简单易用**
   - 无需编写代码
   - 直观的图形界面
   - 适合快速测试和演示

#### Web UI 使用流程

```
1. 启动 Web UI
   └─> ./run_webui.sh

2. 在浏览器打开
   └─> http://localhost:8501

3. 选择配置
   ├─> 选择版本（简化版/全量版）
   └─> 如果选全量版，可配置 RAG/经验库/病例库

4. 输入数据
   ├─> 输入患者信息 (必需)
   ├─> 输入肠鸣音数据 (可选)
   └─> 输入 ECG 数据 (可选)

5. 开始诊断
   └─> 点击"开始诊断"按钮

6. 查看和下载结果
   ├─> 在线查看 diagnosis.json 和 doctor.json
   └─> 下载 JSON 文件保存到本地
```

#### Web UI 截图说明

- **左侧**：输入区域，包含患者信息、肠鸣音、ECG 三个输入框
- **右侧**：诊断结果展示，包含两个标签页（diagnosis.json 和 doctor.json）
- **侧边栏**：配置选项，可以选择版本和知识库设置

### 方法 4: HTTP API 服务器

#### 启动服务器

```bash
# 基本启动
python src/api_server.py

# 自定义端口
python src/api_server.py --port 8080

# 开发模式（热重载）
python src/api_server.py --reload

# 生产环境（多进程）
python src/api_server.py --host 0.0.0.0 --port 8000 --workers 4
```

#### 访问 API 文档

服务器启动后，访问以下地址查看自动生成的 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### API 端点

##### 1. 健康检查

```bash
GET http://localhost:8000/health
```

响应：
```json
{
  "status": "healthy",
  "message": "服务运行正常",
  "expert_count": 28
}
```

##### 2. 患者端诊断接口

```bash
POST http://localhost:8000/api/v1/diagnose/patient
Content-Type: application/json

{
  "patient": {
    "patientName": "张三",
    "patientGender": "男",
    "patientAge": 52,
    "chiefComplaint": "腹痛伴恶心呕吐2天",
    "presentIllness": "患者2天前无明显诱因出现上腹部疼痛..."
  },
  "max_experts": 5
}
```

##### 3. 医生端诊断接口（含风险评估）

```bash
POST http://localhost:8000/api/v1/diagnose/doctor
Content-Type: application/json

{
  "patient": {
    "patientName": "张三",
    "patientGender": "男",
    "patientAge": 52,
    "chiefComplaint": "腹痛伴恶心呕吐2天",
    "presentIllness": "患者2天前无明显诱因出现上腹部疼痛...",
    "clinicCode": "CLINIC_001"
  },
  "bowel_sound": {
    "pred": 0,
    "prob_0": 0.72,
    "prob_1": 0.28
  },
  "ecg": {
    "pred": false,
    "conf": 0.85
  },
  "max_experts": 5
}
```

##### 4. 查询专家池信息

```bash
# 获取所有专家
GET http://localhost:8000/api/v1/experts

# 获取所有专科
GET http://localhost:8000/api/v1/specialties
```

#### 使用 curl 示例

```bash
# 健康检查
curl http://localhost:8000/health

# 诊断请求
curl -X POST http://localhost:8000/api/v1/diagnose/patient \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {
      "patientGender": "男",
      "patientAge": 52,
      "chiefComplaint": "腹痛伴恶心呕吐2天",
      "presentIllness": "患者2天前无明显诱因出现上腹部疼痛，呈阵发性绞痛，向右肩背部放射"
    }
  }'
```

#### 使用 Python requests 示例

```python
import requests
import json

# 服务器地址
API_URL = "http://localhost:8000"

# 患者数据
patient_data = {
    "patient": {
        "patientName": "张三",
        "patientGender": "男",
        "patientAge": 52,
        "chiefComplaint": "腹痛伴恶心呕吐2天",
        "presentIllness": "患者2天前无明显诱因出现上腹部疼痛，呈阵发性绞痛"
    },
    "max_experts": 5
}

# 发送诊断请求
response = requests.post(
    f"{API_URL}/api/v1/diagnose/patient",
    json=patient_data
)

# 解析结果
result = response.json()
print(json.dumps(result, ensure_ascii=False, indent=2))
```

## 📊 性能指标

基于5个验证病例的平均性能：

| 排名 | 科室 | 最佳Fitness | 准确率 | 平均分数 |
|------|------|------------|--------|---------|
| 1 | 肿瘤科 | 0.976 | 100% | 92.0 |
| 2 | 内分泌科 | 0.970 | 100% | 90.0 |
| 3 | 全科医学科 | 0.967 | 100% | 89.0 |
| 4 | 胃肠外科 | 0.961 | 100% | 87.0 |
| 5 | 妇产科 | 0.958 | 100% | 86.0 |
| 6 | 消化内科 | 0.943 | 100% | 81.0 |
| 7 | 呼吸内科 | 0.827 | 80% | 89.0 |
| 8 | 骨科 | 0.809 | 80% | 83.0 |
| 9 | 胸心血管外科 | 0.809 | 80% | 83.0 |
| 10 | 肝胆外科 | 0.773 | 80% | 71.0 |

## 🎓 训练自定义专家池

### 运行完整训练

```bash
# 14科室遗传算法进化
python src/training/run_specialty_evolution.py
```

### 从断点继续训练

```bash
# 从中断处恢复训练
python src/training/continue_specialty_evolution.py
```

### 训练参数配置

```python
TRAINING_CONFIG = {
    "generations": 10,           # 进化代数
    "population_size": 16,       # 种群大小
    "elitism_count": 2,          # 精英保留数量
    "validation_cases": 5,       # 每科室验证病例数
    "top_k_experts": 2,          # 每科室保留专家数
    "early_stopping": 3          # 早停轮数
}
```

## 📝 批量评估

### 顺序批量诊断

```bash
python src/evaluation/batch_diagnosis.py
```

### 并发批量诊断（更快）

```bash
python src/evaluation/batch_diagnosis_concurrent.py
```

## 📁 目录结构

```
.
├── src/                          # 源代码
│   ├── system_step1_route.py     # 步骤1：科室路由
│   ├── system_step2_ir.py        # 步骤2：信息重构
│   ├── system_step3_diag.py      # 步骤3：专家诊断
│   ├── system_step4_agg.py       # 步骤4：结果聚合
│   ├── main_diagnosis_pipeline.py # 主诊断流水线
│   ├── diagnosis_api.py          # CLI诊断接口
│   ├── api_server.py            # HTTP API服务器
│   ├── expert_pool.py            # 专家池管理
│   ├── hybrid_retriever.py       # 混合检索器
│   ├── knowledge_retriever.py    # 知识检索服务
│   ├── training/                 # 训练脚本
│   │   ├── run_specialty_evolution.py
│   │   └── continue_specialty_evolution.py
│   └── evaluation/               # 评估脚本
│       ├── batch_diagnosis.py
│       └── batch_diagnosis_concurrent.py
├── outputs/                      # 输出结果
│   └── optimized_expert_pool_28.json  # 优化的28位专家池
├── data/                         # 数据目录（需自行准备）
├── rag/                          # RAG知识库
├── examples/                     # 示例代码
├── docs/                         # 文档
├── requirements.txt              # Python依赖
├── .gitignore                    # Git忽略配置
├── LICENSE                       # 开源协议
└── README.md                     # 本文件
```

## 🔧 高级配置

### 配置专家池路径

通过环境变量指定专家池文件：

```bash
export EXPERT_POOL_PATH=/path/to/your/expert_pool.json
python src/diagnosis_api.py --patient_file patient.json
```

### 启用 RAG 检索

在初始化时启用 RAG：

```python
pipeline = DiagnosticPipeline(
    activation_mode="eep_semantic",
    enable_rag=True,  # 启用RAG检索
    enable_experience=False,
    enable_case=False
)
```

### 配置 API 参数

编辑 `src/diagnosis_api.py` 中的配置：

```python
API_BASE_URL = "https://your-api-endpoint.com/v1"
API_KEY = "your-api-key"
MODEL_NAME = "gpt-4o"  # 使用的模型
```

## 📖 使用示例

查看 `examples/` 目录获取更多示例：

- `basic_diagnosis_example.py` - 基础诊断示例
- `training_example.py` - 训练示例


