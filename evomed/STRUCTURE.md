# 项目目录结构

```
medical-diagnosis-system/
│
├── src/                              # 源代码
│   ├── __init__.py
│   ├── system_step1_route.py         # 步骤1：科室路由
│   ├── system_step2_ir.py            # 步骤2：信息重构
│   ├── system_step3_diag.py          # 步骤3：专家诊断
│   ├── system_step4_agg.py           # 步骤4：结果聚合
│   ├── main_diagnosis_pipeline.py    # 诊断流水线主程序
│   ├── expert_pool.py                # 专家池管理
│   ├── hybrid_retriever.py           # 混合检索器
│   ├── knowledge_retriever.py        # 知识检索服务
│   │
│   ├── training/                     # 训练模块
│   │   ├── __init__.py
│   │   ├── run_specialty_evolution.py          # 完整训练脚本
│   │   └── continue_specialty_evolution.py     # 断点续训脚本
│   │
│   └── evaluation/                   # 评估模块
│       ├── __init__.py
│       ├── batch_diagnosis.py                  # 批量诊断
│       ├── batch_diagnosis_concurrent.py       # 并发批量诊断
│       └── merge_results.py                    # 结果合并
│
├── outputs/                          # 输出结果
│   ├── optimized_expert_pool_28.json  # 最终优化的28位专家
│   └── expert_pool_initial.json       # 初始专家池
│
├── data/                             # 数据目录（未包含在仓库）
│   └── README.md                      # 数据说明
│
├── rag/                              # RAG知识库
│   ├── rag_build.py                   # RAG索引构建脚本
│   └── 腹痛指南/                      # 医疗指南文档
│
├── examples/                         # 示例代码
│   ├── basic_diagnosis_example.py     # 基础诊断示例
│   └── training_example.py            # 训练示例
│
├── docs/                             # 文档
│   ├── ARCHITECTURE.md                # 架构文档
│   └── QUICK_START.md                 # 快速开始指南
│
├── requirements.txt                  # Python依赖
├── .gitignore                        # Git忽略配置
├── LICENSE                           # 开源协议
├── README.md                         # 项目说明
└── STRUCTURE.md                      # 本文件
```

## 核心文件说明

### 诊断流程
- `system_step1_route.py`: 科室路由，确定患者应就诊科室
- `system_step2_ir.py`: 信息重构，将病历结构化
- `system_step3_diag.py`: 专家诊断，调用专科专家进行推理
- `system_step4_agg.py`: 结果聚合，整合多专家意见

### 专家池管理
- `expert_pool.py`: 可演化专家池（EEP）的管理和激活
- `main_diagnosis_pipeline.py`: 集成四步流程的主程序

### 知识增强
- `knowledge_retriever.py`: 统一知识检索接口
- `hybrid_retriever.py`: RAG/经验库/病例库混合检索

### 训练脚本
- `run_specialty_evolution.py`: 14个科室的完整遗传算法训练
- `continue_specialty_evolution.py`: 从中断处继续训练

### 评估脚本
- `batch_diagnosis.py`: 批量诊断评估
- `batch_diagnosis_concurrent.py`: 并发批量诊断（更快）

### 输出文件
- `optimized_expert_pool_28.json`: 经过遗传算法优化的28位专家
  - 14个科室 × 2位专家/科室
  - 包含优化后的诊断提示词
  - 附带性能指标（fitness, accuracy, avg_score）

## 不包含在仓库中的文件

以下文件因体积或隐私原因未包含在Git仓库中（见 `.gitignore`）：

- `venv/`: Python虚拟环境
- `__pycache__/`: Python缓存
- `*.log`: 日志文件
- `data/*.xlsx`: 患者数据（敏感信息）
- `ga_gen_*_checkpoint.json`: 训练中间检查点
- `specialty_pool_*.json`: 各科室中间结果
- `rag/rag_index/`: RAG向量索引（较大）
- `exp/`: 实验性代码

## 文件大小参考

- 源代码: ~50KB
- 优化专家池: ~150KB
- RAG索引: ~100MB（如启用）
- 训练日志: ~10MB（完整训练）
