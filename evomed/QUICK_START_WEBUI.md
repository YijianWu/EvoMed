# Web UI 快速启动指南

## 🚀 5分钟快速上手

### 第一步：安装依赖

```bash
# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装所需包
pip install -r requirements.txt
```

### 第二步：配置 API

编辑 `src/diagnosis_api.py`（约 295-297 行），配置您的 API 密钥：

```python
API_BASE_URL = "https://your-api-endpoint.com/v1"
API_KEY = "your-api-key-here"
MODEL_NAME = "gpt-4o"
```

### 第三步：启动 Web UI

```bash
# 使用启动脚本（推荐）
./run_webui.sh

# 或者直接使用 streamlit
streamlit run src/web_ui.py --server.port 8501
```

### 第四步：访问界面

在浏览器中打开：**http://localhost:8501**

### 第五步：开始诊断

1. **点击"加载示例数据"** - 快速加载测试数据
2. **点击"开始诊断"** - 执行诊断分析
3. **查看结果** - 在右侧查看 diagnosis.json 和 doctor.json
4. **下载文件** - 点击下载按钮保存结果

---

## 📋 数据格式说明

### 患者信息 (patient.json)

必填字段：
- `patientGender`: 性别（"男" 或 "女"）
- `patientAge`: 年龄（数字）
- `chiefComplaint`: 主诉
- `presentIllness`: 现病史

选填字段：
- `patientName`: 姓名
- `personalHistory`: 既往史
- `labs`: 检验指标（数组）
- `exam`: 检查指标（数组）
- `clinicCode`: 诊所代码

### 肠鸣音数据 (c.json) - 可选

```json
{
  "fold": "fold_0",
  "pid": "患者ID",
  "pred": 0,        // 0=正常, 1=异常
  "prob_0": 0.72,   // 正常概率
  "prob_1": 0.28    // 异常概率
}
```

### ECG 数据 (ecg.json) - 可选

```json
{
  "pid": "患者ID",
  "pred": false,    // false=正常, true=异常
  "conf": 0.85,     // 置信度
  "topk": [["False", 0.85], ["True", 0.15]]
}
```

---

## ⚙️ 版本选择

### 简化版（默认）

✅ **优点**：
- 启动快速（约 10 秒）
- 诊断速度快（约 1-2 分钟）
- 资源占用少
- 适合快速测试

❌ **限制**：
- 不包含 RAG 检索
- 不包含经验库检索
- 不包含病例库检索

### 全量版

✅ **优点**：
- 功能完整
- 支持 RAG 医学指南检索
- 支持经验库检索（A-Mem）
- 支持病例库检索（ACE）
- 诊断更准确全面

❌ **限制**：
- 启动较慢（约 30-60 秒）
- 诊断速度较慢（约 3-5 分钟）
- 需要更多系统资源
- 需要配置知识库路径

**如何启用全量版**：
1. 在侧边栏勾选 "使用全量版本"
2. 选择需要启用的知识库（RAG/经验库/病例库）
3. 点击"开始诊断"

---

## 🔧 常见问题

### 1. 端口被占用

如果 8501 端口已被使用，修改启动命令：

```bash
streamlit run src/web_ui.py --server.port 8502
```

### 2. 无法访问（远程服务器）

确保防火墙开放端口，或使用 SSH 隧道：

```bash
ssh -L 8501:localhost:8501 user@your-server
```

### 3. API 调用失败

检查：
1. API 密钥是否正确
2. 网络连接是否正常
3. API 额度是否充足

### 4. 全量版无法启动

全量版需要以下文件/目录：
- `outputs/optimized_expert_pool_28.json` - 专家池文件
- `expert_pool.json` - EEP 专家池
- `./rag/rag_index/` - RAG 索引（如启用）
- `./exp/A-mem-sys/memory_db/` - 经验库（如启用）

如果缺少这些文件，请：
1. 使用简化版
2. 或者运行训练脚本生成这些文件

### 5. 诊断速度慢

优化建议：
- 使用简化版（不启用 RAG/经验库/病例库）
- 减少激活的专家数量
- 使用更快的 API 模型

---

## 📊 输出说明

### diagnosis.json

包含完整的诊断信息：
- `status`: 诊断状态
- `patient_info`: 患者可见信息
  - `diagnosis_top5`: 前5个诊断
  - `diagnosis_basis`: 诊断依据
  - `differential_diagnosis`: 鉴别诊断
  - `suggested_examinations`: 建议检查
- `risk_assessment`: 风险评估（医生专属）
  - `risk_level`: 危险分层
  - `visit_advice`: 就诊建议
  - `emergency_needed`: 是否需要急救
- `expert_opinions`: 专家意见列表

### doctor.json

医生端/管理端展示格式：
- `type`: 消息类型
- `summary_value`: 总结
- `diagnostic_result_value`: 诊断结果
- `condition_analysis_value`: 病情分析
- `suggestions_examinations_value`: 检查建议
- `clinic_code`: 诊所代码

---

## 🎯 使用建议

1. **首次使用**：先点击"加载示例数据"测试功能
2. **实际诊断**：使用简化版即可满足大部分需求
3. **学术研究**：使用全量版获取更详细的分析
4. **批量处理**：建议使用 CLI 或 HTTP API
5. **演示展示**：Web UI 最直观易用

---

## 📞 获取帮助

如有问题，请：
1. 查看完整 README.md
2. 查看项目文档（docs/ 目录）
3. 提交 Issue
4. 联系项目维护者

---

**祝使用愉快！🎉**


