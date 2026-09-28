# MedicalGPT 7 天学习与简历项目规划

> 项目暂定名：**MedGuardGPT——中文临床病例结构化分析与安全对齐系统**  
> 学习时间：**2026 年 9 月 17 日—9 月 23 日**  
> 每日投入：**8–10 小时**  
> 可用算力：**8×RTX 3090（24GB）**  
> 当前基础：会 Python，大模型微调经验较少  
> 项目用途：学习、研究和求职展示，不用于真实诊疗

---

## 1. 最终目标

七天结束时，不只是“运行过 MedicalGPT”，而是形成一个能写入简历、能演示、能复现、能经得住面试追问的完整项目：

```text
公开医疗数据
→ 清洗、去重和质量分析
→ 面向目标任务的数据筛选
→ 构造结构化 SFT 数据
→ LoRA-SFT
→ 构造医疗安全偏好对
→ DPO 安全对齐
→ 通用基准 + 自建测试集评测
→ 消融实验与错误分析
→ API/Demo
→ README、实验报告、简历和面试材料
```

项目集中解决两个问题：

1. 通用模型面对中文病例时，容易出现格式不稳定、诊断依据不完整、输入证据利用不足等问题。
2. 医疗问答中容易出现过度确诊、虚构检查结果、遗漏危险信号、无依据提供药物或剂量等安全问题。

最终项目名称建议使用：

> **MedGuardGPT：基于 Qwen3.5-4B 的中文临床病例结构化分析与安全对齐系统**

---

## 2. 对三个参考项目的吸收与改造

### 2.1 MedicalGPT：训练框架底座

参考：[MedicalGPT](https://github.com/shibing624/MedicalGPT)

主要学习和复用：

- ShareGPT/JSONL 数据格式；
- Qwen 系列的对话模板和 tokenizer 处理；
- LoRA/QLoRA SFT；
- DPO 偏好优化；
- LoRA 合并、推理和服务化；
- 多卡训练、梯度累积和显存优化；
- TensorBoard、checkpoint、断点恢复。

本项目不会把 PT、SFT、RM、PPO、DPO、GRPO 全部机械运行一遍。七天主线只执行：

```text
Base Model → SFT → DPO → Evaluation → Deployment
```

PT/CPT、RM/PPO、GRPO 只学习原理。这样可以把时间放在数据、评测和实验依据上。

### 2.2 HealthAI-2025：数据构造思路

参考：[HealthAI-2025](https://github.com/yuandaxia2001/HealthAI-2025)

可借鉴的主线：

```text
原始医疗问答
→ 重构成目标病例格式
→ 对目标集和候选集向量化
→ 按相似度筛选训练数据
→ 使用强模型生成高质量目标答案
→ LoRA 微调小模型
```

不能直接照搬的部分：

- 不能只做相似度排序而不检查重复和数据泄漏；
- 不能把强模型生成内容直接当作医学金标准；
- 不能只展示最终案例，没有基线和消融；
- 不蒸馏冗长的隐藏思维链，改为输出可核验的简洁证据；
- 增加医疗安全偏好数据、危险信号评测和能力回退检查；
- 记录 API 模型版本、提示词、失败率、token 和成本。

### 2.3 lm-evaluation-harness：统一评测

参考：[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

主要使用方式：

- 统一运行 C-Eval、CMMLU 医学子任务；
- 保存任务版本、提示词、生成参数和原始结果；
- 使用 YAML 或外部任务目录接入自建评测集；
- 保证 Base、SFT、DPO 使用相同评测条件；
- 评测结果保留逐样本输出，支持 bad case 分析。

---

## 3. 项目亮点设计

### 3.1 亮点一：目标分布感知的数据筛选

从约 5 万条候选医疗问答中，依次完成：

1. 格式与长度过滤；
2. 广告、乱码、模板污染和隐私信息处理；
3. 精确哈希去重；
4. MinHash/LSH 近似去重；
5. 句向量语义去重；
6. 与目标病例画像的 embedding 相似度计算；
7. 类别配额和聚类多样性约束；
8. 筛选约 4,000 条高价值 SFT 数据。

同时随机抽取同样数量的数据作为对照，以证明提升是否来自数据质量。

### 3.2 亮点二：可验证的结构化病例分析

模型不输出无法核验的长篇思维过程，而输出：

- 可能诊断；
- 确定性等级；
- 来自输入病例的依据；
- 缺失信息；
- 危险信号；
- 分诊建议；
- 医疗免责声明。

这样可以直接计算格式合法率、证据支持率、危险信号召回率和分诊准确率。

### 3.3 亮点三：医疗安全 DPO

构造约 1,000 组 chosen/rejected 偏好对，集中处理：

- 过度确诊；
- 虚构症状、检查或病史；
- 信息不足时仍给出确定结论；
- 漏掉急症危险信号；
- 无依据提供处方或具体剂量；
- JSON 格式破坏；
- 过度拒答。

### 3.4 亮点四：完整实验矩阵和消融

| 实验 | 模型 | 目的 |
|---|---|---|
| E0 | 原始 Qwen3.5-4B | 基线 |
| E1 | 随机 4K 数据 SFT | 判断“单纯训练”的收益 |
| E2 | 清洗和目标筛选 4K 数据 SFT | 证明数据工程价值 |
| E3 | E2 + 医疗安全 DPO | 证明偏好对齐价值 |

E1 和 E2 必须使用相同样本数、训练步数、超参数和随机种子。

---

## 4. 数据与模型接口

### 4.1 输入病例格式

```json
{
  "case_id": "case_000001",
  "case_text": "患者主诉、现病史、既往史和检查结果",
  "source": "huatuo26m_lite",
  "category": "respiratory"
}
```

### 4.2 模型目标输出

```json
{
  "possible_diagnoses": [
    {
      "name": "疾病名称",
      "certainty": "high|medium|low"
    }
  ],
  "evidence": [
    "只能引用输入病例中存在的症状、体征或检查"
  ],
  "missing_information": [
    "影响判断但输入未提供的信息"
  ],
  "red_flags": [
    "需要紧急关注的危险表现"
  ],
  "triage": "emergency|urgent|routine|self_care",
  "safety_notice": "研究用途，不替代医生诊断"
}
```

### 4.3 SFT 数据格式

```json
{
  "conversations": [
    {"from": "system", "value": "系统约束"},
    {"from": "human", "value": "病例文本"},
    {"from": "gpt", "value": "结构化 JSON"}
  ]
}
```

### 4.4 DPO 数据格式

```json
{
  "conversations": [
    {"from": "system", "value": "系统约束"},
    {"from": "human", "value": "病例文本"}
  ],
  "chosen": "安全、依据充分、格式正确的回答",
  "rejected": "过度确诊、虚构依据或遗漏危险信号的回答"
}
```

### 4.5 数据划分原则

- 不能逐行随机划分后就直接训练；
- 应先进行去重和语义聚类，再按语义簇划分 train/dev/test；
- 同一疾病模板、同义改写或近似病例不能跨集合；
- C-Eval、CMMLU 不得进入训练或筛选；
- 测试集在最终训练前冻结并记录哈希。

---

## 5. 推荐训练配置

### 5.1 主模型

- 模型：`Qwen/Qwen3.5-4B`；
- 方法：BF16 LoRA；
- 不优先使用 QLoRA，因为 8×3090 足以支撑 4B LoRA，且 BF16 调试更直观；
- 上下文长度：1024；
- LoRA rank：16；
- LoRA alpha：32；
- LoRA dropout：0.05；
- target modules：全部线性层。

### 5.2 SFT

- 每卡 batch size：2；
- gradient accumulation：4；
- 8 卡全局 batch：64；
- epoch：3；
- learning rate：`2e-5`；
- scheduler：cosine；
- warmup ratio：0.03；
- weight decay：0.01；
- gradient checkpointing：开启；
- 先用 100 条数据和单卡完成 smoke test；
- 再验证 2 卡 NCCL；
- 最后运行 8 卡正式训练。

### 5.3 DPO

- 先合并最佳 SFT LoRA；
- 从合并后的 SFT 模型启动新的 DPO LoRA；
- 偏好数据：约 1,000 对；
- max source length：768；
- max target length：512；
- 每卡 batch size：1；
- gradient accumulation：4；
- 全局 batch：32；
- learning rate：`5e-7`；
- beta：0.1；
- epoch：2。

### 5.4 多卡故障处理顺序

1. 单卡前向和反向；
2. 单卡保存、加载和合并 LoRA；
3. 两卡 DDP；
4. 八卡正式训练；
5. 八卡 NCCL 不稳定时改用四卡并增加梯度累积，保持全局 batch；
6. 只有显存实测不足时才引入 DeepSpeed ZeRO-2；
7. 不为展示技术名词强行使用 ZeRO-3。

---

## 6. 评测体系

### 6.1 外部医学知识任务

- `ceval-valid_basic_medicine`；
- `ceval-valid_clinical_medicine`；
- `ceval-valid_physician`；
- `cmmlu_clinical_knowledge`；
- `cmmlu_college_medicine`。

### 6.2 自建测试集

结构化病例测试集：400 条。

医疗安全测试集：300 条。

- 100 条急症或明显危险信号；
- 80 条信息不足、不能直接确诊；
- 60 条诱导具体处方或危险剂量；
- 30 条信息相互矛盾；
- 30 条非医疗或分布外输入。

### 6.3 核心指标

- JSON Valid Rate；
- Schema Completion Rate；
- Diagnosis Recall@3；
- Evidence Support Rate；
- Unsupported Claim Rate；
- Red-flag Recall；
- Triage Accuracy；
- Unsafe Advice Rate；
- Appropriate Abstention Rate；
- C-Eval/CMMLU Accuracy；
- 首 token 延迟、总延迟、吞吐和显存峰值。

### 6.4 目标与验收线

- E2/E3 的 JSON Valid Rate ≥98%；
- E2 的结构完整率和证据支持率明显优于 E0、E1；
- E3 相比 E2 的 Unsafe Advice Rate 相对下降至少 30%；
- E3 的 Red-flag Recall 高于 E2；
- DPO 后外部医学知识平均分下降不超过 2 个百分点；
- 所有指标保留逐样本预测并计算 bootstrap 95% 置信区间；
- 未达到目标时如实报告，使用 bad case 解释原因，不筛选性展示结果。

---

## 7. 七天详细安排

## Day 1：9 月 17 日——理解仓库、跑通链路、建立基线

### 上午：仓库和源码

- 阅读 README、SFT、DPO、template、数据加载、LoRA 合并和推理入口；
- 画出 JSONL → tokenizer → input_ids/labels → loss → checkpoint 的调用链；
- 确认 prompt/answer 拼接方式；
- 确认用户 token 是否被 `-100` mask；
- 理解 chat template、BOS/EOS/PAD；
- 检查训练集和验证集不能指向同一个目录。

### 下午：最小训练闭环

- 建立独立环境并固定依赖版本；
- 下载 Qwen3.5-4B；
- 完成原模型推理；
- 使用 100 条样本、单卡、1 epoch 运行 SFT smoke test；
- 验证 checkpoint、LoRA 合并和合并后推理；
- 使用 lm-evaluation-harness 对医学任务运行 `--limit` 小样本。

### 晚上：基线与问题定义

- 对 E0 运行完整外部医学知识基线；
- 固定随机种子、环境、命令、GPU 信息和生成参数；
- 写出项目问题、两个核心假设和实验矩阵。

### 当日交付

- 仓库结构和训练数据流图；
- smoke-test 命令；
- E0 基线；
- 环境与故障记录。

### 退出条件

能够解释一条训练样本如何变成 loss，并能排查模板、截断、OOM 或空 label 问题。

## Day 2：9 月 18 日——数据清洗、去重和目标筛选

### 上午：数据审计

- 下载并查看 Huatuo26M-Lite；
- 固定抽取 50,000 条候选数据；
- 分析字段、长度、类别、乱码、广告、模板和重复；
- 完成格式、隐私、长度和质量规则；
- 统计每项规则的淘汰比例。

### 下午：去重与向量筛选

- 学习并实现哈希去重、MinHash、LSH 和 Jaccard；
- 使用句向量处理语义重复；
- 建立 500 条目标画像开发集；
- 向量化目标画像和候选数据；
- 计算 top-k 平均相似度；
- 加入类别配额、聚类或 MMR 多样性约束；
- 选出 4,000 条精选数据；
- 固定随机抽取 4,000 条作为 E1。

### 晚上：数据报告

- 检测 train/dev/test 语义泄漏；
- 绘制数量漏斗、长度、类别、重复率和相似度分布；
- 抽检入选和淘汰样本各 30 条。

### 当日交付

- 数据流水线；
- 随机集和精选集；
- 数据质量报告；
- 数据血缘和 split manifest。

### 退出条件

能够解释每条数据为什么入选或被淘汰，并能证明没有明显数据泄漏。

## Day 3：9 月 19 日——构造 SFT 和 DPO 数据

### 上午：提示词与结构协议

- 设计 system prompt 和 JSON Schema；
- 先生成 50 条试验数据；
- 统计 JSON 失败、无依据结论、漏掉红旗、输出过长和字段不一致；
- 调整提示词，使首次 JSON 通过率达到 95% 以上。

### 下午：批量构造 SFT 数据

- 批量转换 4,000 条精选样本；
- 每项失败最多自动重试两次；
- 仍失败的数据进入 quarantine；
- 校验 schema、枚举、证据引用和长度；
- 抽查 100 条；
- 记录 API 模型、提示词版本、token、费用和失败率；
- 转换为 MedicalGPT ShareGPT 格式。

### 晚上：偏好数据

- 构造约 1,000 对 chosen/rejected；
- 覆盖过度确诊、虚构依据、危险用药、红旗遗漏、格式错误；
- 控制 chosen/rejected 长度接近，避免长度偏好；
- 人工抽查至少 100 对。

### 当日交付

- 4K SFT 数据；
- 1K DPO 数据；
- schema validator；
- API 成本和失败率报告；
- dataset card 草稿。

### 退出条件

任意样本都能追溯原始来源、清洗规则、生成提示词和最终 split。

## Day 4：9 月 20 日——SFT 学习与训练

### 上午：必须掌握的训练原理

- causal LM next-token loss；
- label masking；
- LoRA 低秩更新；
- rank、alpha、dropout；
- batch、梯度累积和全局 batch；
- BF16、FP16、QLoRA/NF4；
- gradient checkpointing；
- DDP 梯度同步。

### 下午至晚上：正式 SFT

- 先验证两卡；
- 运行 E1 随机数据 SFT；
- 运行 E2 精选数据 SFT；
- 监控 loss、learning rate、grad norm、tokens/s、GPU 利用率和显存；
- 使用固定 20 条病例进行阶段性对比；
- 发现异常时优先检查模板、截断率、空 label 和学习率。

### 当日交付

- E1、E2 LoRA；
- TensorBoard 曲线；
- 训练配置和资源报告；
- 初步案例对比。

### 退出条件

能够说明 loss 下降为什么不等于业务能力提升，并解释 E1/E2 差异。

## Day 5：9 月 21 日——SFT 评测、错误分析和 DPO

### 上午：SFT 对比

- 对 E0、E1、E2 运行同一套评测；
- 比较数据筛选收益和通用能力回退；
- 将错误分成诊断错误、证据不支持、信息遗漏、红旗遗漏、格式错误、过度确定和过度拒答。

### 下午：DPO 原理和 smoke test

- 合并 E2 LoRA；
- 学习 policy/reference、chosen/rejected、log-ratio 和 beta；
- 理解 DPO 与 RM+PPO、GRPO 的区别；
- 使用 50 对数据运行 DPO smoke test；
- 检查 reward margin 是否朝正确方向变化。

### 晚上：正式 DPO

- 运行 E3；
- 对固定安全病例比较训练前后行为；
- 检查是否出现全面拒答；
- 如出现过度拒答，调整正常病例比例或 DPO 强度。

### 当日交付

- E0/E1/E2 中期结果；
- bad case 分类；
- E3 adapter；
- DPO loss、reward margin 和安全案例。

### 退出条件

能够解释为什么选择 DPO，并说明不执行 PPO/GRPO 的项目取舍。

## Day 6：9 月 22 日——完整评测和消融

### 上午：统一评测

- 对 E0–E3 运行全部外部任务、自建病例和安全集；
- 保存原始生成结果；
- 计算 bootstrap 95% 置信区间。

### 下午：核心消融

1. 随机数据 vs 目标筛选数据；
2. Base vs SFT；
3. SFT vs SFT+DPO；
4. 可选：规则清洗 vs 清洗+语义去重。

### 晚上：错误分析

- 人工检查 50–100 条结果；
- 统计失败类别；
- 选择 6–10 个代表性成功和失败案例；
- 明确哪些指标改善、哪些未改善、是否存在能力回退。

### 当日交付

- 最终结果表和图；
- 消融结果；
- 置信区间；
- bad-case 报告；
- 已知限制。

### 退出条件

每项亮点都有表格、曲线或案例支持，而不是主观描述。

## Day 7：9 月 23 日——部署、文档、简历和面试

### 上午：部署

- 使用 vLLM 或 Transformers 部署 E3；
- 提供 `/health` 和 `/analyze`；
- 制作简单 Gradio 页面；
- 展示结构化输出、危险信号和免责声明；
- 测量吞吐、首 token 延迟和显存。

### 下午：项目文档

- 完成问题定义、架构图、数据流水线、训练配置、评测、结果、消融和 bad case；
- 完成一键复现命令；
- 完成 dataset card、model card 和实验报告；
- 清除硬编码路径、API key 和临时文件；
- 给 schema、数据划分和评测指标补测试。

### 晚上：求职包装

- 从真实结果生成简历描述；
- 准备 3 分钟和 10 分钟项目介绍；
- 模拟至少 20 个面试问题；
- 固定代码、配置和结果版本。

### 当日交付

- Demo/API；
- README；
- 数据集卡和模型卡；
- 实验报告；
- 简历描述；
- 面试问答稿；
- 可复现 release。

---

## 8. 必须掌握的知识点

### 8.1 Transformer 与因果语言模型

- embedding、位置编码；
- Q/K/V、scaled dot-product attention；
- multi-head attention；
- causal mask；
- MLP、残差、RMSNorm；
- next-token prediction 和交叉熵；
- KV Cache；
- greedy、temperature、top-p。

### 8.2 Tokenizer 和对话模板

- BPE/SentencePiece；
- token、字符和词的关系；
- BOS/EOS/PAD；
- `apply_chat_template`；
- padding side、truncation；
- assistant-only loss；
- 模板错误、乱码和停止失败的原因。

### 8.3 数据工程

- JSON、JSONL、ShareGPT；
- schema validation；
- 哈希、MinHash、LSH、Jaccard；
- embedding、cosine、top-k、MMR；
- 数据质量、类别平衡；
- 数据泄漏和 benchmark contamination；
- API 并发、重试、断点续跑和幂等；
- 数据版本、血缘、token 和成本统计。

### 8.4 LoRA/QLoRA 和训练

- 低秩矩阵更新；
- rank、alpha、dropout、target modules；
- 全参、LoRA、QLoRA 的差异；
- FP32、FP16、BF16、INT8、NF4；
- batch、gradient accumulation、epoch、step；
- learning rate、warmup、weight decay；
- gradient clipping/checkpointing；
- OOM、NaN、loss 不降和过拟合排查。

### 8.5 分布式训练

- 数据并行和模型并行；
- DDP 梯度同步；
- global batch 计算；
- `torchrun`、local rank、world size；
- NCCL 常见故障；
- DeepSpeed ZeRO-1/2/3；
- 为什么本项目默认不需要 ZeRO-3。

### 8.6 SFT、DPO、RLHF

- SFT 目标和局限；
- chosen/rejected；
- policy/reference model；
- DPO beta 和隐式奖励；
- RM+PPO 和 DPO 的差别；
- GRPO 的 group-relative reward；
- 奖励投机、过度拒答和灾难性遗忘。

### 8.7 评测

- closed-set 和 open-ended 评测；
- accuracy、EM、F1、Recall@k；
- JSON 合法率、证据支持率、安全率；
- LLM judge 的位置、长度和自偏好；
- 人工 rubric；
- bootstrap 置信区间；
- 控制变量和消融；
- 为什么 loss 不是最终指标。

### 8.8 工程与部署

- checkpoint、adapter、merge；
- TensorBoard；
- seed 和配置管理；
- vLLM continuous batching；
- FastAPI 校验和错误处理；
- 单元测试和集成测试；
- API key、隐私保护；
- model card、dataset card 和免责声明。

### 8.9 医疗安全

- 不把项目描述成医生替代品；
- 急症、信息不足、药物剂量和特殊人群的安全边界；
- 输入证据和输出结论可追溯；
- 公开医疗问答中的错误不能直接作为金标准；
- 数据去标识化和最小化使用。

---

## 9. 七天内只理解、不执行的内容

- PT/CPT：理解领域知识注入、PPL 和灾难性遗忘；
- RM+PPO：理解 RLHF 三阶段和奖励模型偏差；
- GRPO：理解 group-relative advantage 和可验证奖励；
- 自训练 tokenizer、扩词表；
- LongRoPE、mid-training 和长上下文；
- RAG：作为二期项目；
- 从零训练 BERT 数据质量分类器；
- 7B 全参数微调。

这样做不是知识缺失，而是为了让七天项目拥有清晰主线和可信证据。

---

## 10. 项目验收标准

项目只有同时满足以下条件才算完成：

- 原始数据能够一键生成固定版本的训练集；
- E0、E1、E2、E3 结果齐全；
- 至少完成三项核心消融；
- 关键指标能追溯到逐样本预测；
- Demo 能对新病例返回合法结构；
- README 可以让他人在新环境复现；
- 数据来源、许可证、限制和医疗风险清楚；
- 简历中的每个数字都与实验记录一致。

重点测试：

- 合法病例可以生成完整字段；
- 缺失病史时列出 missing information；
- 明显急症的分诊为 emergency/urgent；
- 诱导开药时不输出无依据剂量；
- 信息矛盾时不强行确诊；
- 非医疗问题不伪造医学结论；
- 超长、空文本、特殊字符有确定行为；
- 数据 split 没有高相似样本泄漏；
- LoRA 合并前后固定样本结果基本一致；
- Base/SFT/DPO 使用相同评测参数。

---

## 11. 简历项目描述模板

所有 `[X]` 必须在实验完成后替换成真实结果：

> **MedGuardGPT｜中文临床病例结构化分析与安全对齐系统**
>
> - 基于 Qwen3.5-4B 与 MedicalGPT 搭建 SFT→DPO 训练链路，使用 8×RTX 3090 完成多卡 LoRA 微调及可复现实验管理。
> - 设计医疗数据清洗、MinHash/语义去重、目标分布相似度筛选与多样性约束流水线，将 5 万条候选数据筛选为 4 千条高质量结构化病例；相较同规模随机数据，`[核心指标]` 提升 `[X]`。
> - 构造 1 千组医疗安全偏好对并进行 DPO，对过度诊断、虚构依据、危险用药和红旗遗漏进行对齐，使 Unsafe Advice Rate 下降 `[X]%`，Red-flag Recall 提升 `[X]`。
> - 基于 lm-evaluation-harness 和自建 700 条病例/安全测试集，完成 Base、SFT、DPO 及数据筛选消融；通过 FastAPI/vLLM 部署并达到 `[X] tokens/s`。

禁止使用以下表述：

- “自主研发医疗大模型”；
- “达到医生水平”；
- “医学准确率 99%”；
- “完成 PT/SFT/RLHF/GRPO 全链路”，除非确实执行；
- 找不到原始实验记录的提升数字。

---

## 12. 面试问题清单

至少能够回答：

1. 为什么选择 MedicalGPT，而不是 LLaMA-Factory？
2. 为什么选择 4B，而不是直接训练 7B？
3. 数据相似度筛选是否造成测试集泄漏？
4. MinHash 和 embedding 去重分别解决什么问题？
5. 为什么不能直接使用 Huatuo 原回答作为金标准？
6. 如何证明提升来自数据质量，而不是训练步数？
7. LoRA rank 越大一定越好吗？
8. SFT 为什么可能损害通用能力？
9. DPO 的 beta 控制什么？
10. DPO 后为什么可能过度拒答？
11. 为什么不执行 PPO/GRPO？
12. 开放式诊断回答如何评测？
13. LLM judge 有哪些偏差？
14. 全局 batch size 如何计算？
15. 8 卡 DDP 为什么每张卡仍需要放一份模型？
16. BF16 和 FP16 有什么区别？
17. 模型输出非法 JSON 如何处理？
18. 医疗大模型最重要的安全边界是什么？
19. 如果诊断指标没有提高，项目还有什么价值？
20. 下一阶段如何加入医生标注、RAG 或真实业务数据？

---

## 13. 风险与降级方案

### API 数据生成成本超出预算

- 保留 4K 数据筛选结果；
- 只生成 2K–3K 高质量数据；
- 优先覆盖安全集和核心病种；
- 在报告中记录实际规模，不声称使用 4K。

### 8 卡训练不稳定

- 使用 4 卡；
- 增加 gradient accumulation 保持全局 batch；
- 记录硬件和通信问题；
- 不临时切换复杂分布式框架。

### SFT 没有提升医学知识分数

- 检查 tokenizer/template、label mask 和截断；
- 判断训练目标是否只改善结构而非知识；
- 如实把结论改为“结构与安全对齐”；
- 不通过筛选测试样本制造虚假提升。

### DPO 造成过度拒答

- 增加正常、低风险病例 chosen；
- 降低 beta 或训练步数；
- 检查 chosen/rejected 长度偏差；
- 同时报告安全收益与可用性损失。

### 缺少医生评审

- 不声称专家评估；
- 使用公开考试基准、规则指标、双模型裁判和人工抽查；
- 将“临床专家盲评”列为二期工作。

---

## 14. 后续二期方向

七天项目完成后再选择一个方向深化：

1. 加入循证医学知识库和 RAG，减少事实性错误；
2. 请医学专业人员标注 200–500 条高质量测试集；
3. 加入置信度校准和 selective prediction；
4. 将规则安全检查扩展为独立 guardrail；
5. 比较 DPO、ORPO、GRPO 的安全对齐效果；
6. 训练 7B 模型并比较参数规模与数据质量的收益；
7. 将数据选择扩展为主动学习或 influence-based selection。

---

## 15. 执行原则

1. 先保证最小闭环，再扩展实验。
2. 每天都必须产生可检查的交付物。
3. 所有训练前先进行小数据 smoke test。
4. 所有结果必须保存原始预测和配置。
5. 不直接复制别人的项目描述、数据成品或实验结果。
6. 不把 loss 曲线当作最终成果。
7. 不为了“技术栈丰富”加入无关模块。
8. 不伪造指标、专家评估或临床价值。
9. 项目亮点必须能被消融实验验证。
10. 最终目标是能够独立解释、复现和排错，而不是只会运行 shell 脚本。
