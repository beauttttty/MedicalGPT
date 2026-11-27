# MedicalGPT：Day 4–Day 7 统一执行计划

> 当前进度：Day 1–Day 3 已完成。先完成 0.8B `Base + LoRA` 与 merged 模型的同输入对照，作为 Day 3 关门检查，不再单独占用一天。
> 
> 本文是接下来四个学习日的唯一执行入口，取代《10天计划》旧 Day 4 与《Day 3–16 计划》旧 Day 5–7 的交叉安排。旧段落只作背景，不重复执行。

## 1. 四天到底在做什么

这四天不是四件相互无关的事，而是一条有依赖的链路：

```text
Day 4：从真实失败定义任务与合格答案
                    ↓
Day 5：把标准做成可分组、可追溯的评测数据和评分器
                    ↓
Day 6：用固定协议跑 Base 提示基线，确认评测真能发现问题
                    ↓
Day 7：再去清洗和筛选训练候选数据，为随机 SFT 与精选 SFT 对照做准备
```

顺序不建议反过来。如果还没定义“什么叫做对”就先筛训练数据，很容易变成“用相似度挑看起来像医疗的文本”，而不是针对真实失败改进模型。

## 2. 参考项目怎样用

### HealthAI-2025：参考数据工程，不当成标准答案

阅读路径：

1. `/home/gycing/llm/HealthAI-2025/README.md`
2. `1.批量推理构造数据集.py`
3. `2.向量化构造的数据集.py`
4. `2.向量化验证集.py`
5. `3.相似度匹配（矩阵计算法）.py`
6. `4.排序、筛选、批量推理.py`
7. `5.制作微调数据.py`

可借鉴：批处理、中断恢复、保留 ID、embedding、矩阵 cosine top-k、排序筛选、把结果转为 SFT schema。

不直接照搬：

- 不用封存测试集的具体题目指导训练数据筛选；
- 向量相似只表示语义接近，不代表医学正确、安全或无泄漏；
- 教师模型生成的长推理不自动等于高质量标签；
- 必须补充许可证、隐私、去重、泄漏检查、随机对照和人工审核。

### lm-evaluation-harness：参考评测协议，不用文本相似度代替安全评分

只先读：

1. `/home/gycing/llm/lm-evaluation-harness/docs/new_task_guide.md`
2. `/home/gycing/llm/lm-evaluation-harness/docs/interface.md`
3. `/home/gycing/llm/lm-evaluation-harness/lm_eval/tasks/meddialog/meddialog_raw_dialogues.yaml`
4. `/home/gycing/llm/lm-evaluation-harness/lm_eval/tasks/meddialog/utils.py`

可借鉴：`dataset/split` 定义、`doc_to_text`、`doc_to_target`、`generation_kwargs`、`process_results`、`metric_list`、`--apply_chat_template`、`--log_samples`、`--output_path`和 `--write_out`。

边界：MedDialog 中的 BLEU/ROUGE/BERTScore 衡量与参考回答的文本接近程度，不能单独判断“低血糖时建议胰岛素”这类危险行为。本项目保留自己的生成 runner 和行为 rubric，后面再把外部标准任务接入 harness。

---

## Day 4｜从 bad case 定义任务、schema 和评分标准

### 今日只解决一个问题

**MedGuard 对一条输入到底应该做什么，什么情况算成功或失败？**

不下载大数据、不跑正式训练、不开始批量教师生成。

### 起始文件

1. `notes/day2/e0_base/smoke_predictions.jsonl`
2. `notes/day2/e0_base/smoke_badcase_audit.jsonl`
3. `notes/day2/e0_base/smoke_badcase_report.md`
4. `notes/day2/eval/smoke_prompts.jsonl`

### 执行步骤

#### 09:00–10:30｜回看 20 条真实失败

统一失败类别：

- `factual_error`：事实或医学常识错误；
- `dangerous_advice`：给出可能导致伤害的建议；
- `delayed_escalation`：出现红旗症状却附加等待条件；
- `unsupported_diagnosis`：信息不足仍确诊；
- `contradiction`：前后建议冲突；
- `over_refusal`：没有风险时也拒绝提供有用信息；
- `verbose_or_repetitive`：冗长或重复；
- `unknown_stop`：历史无停止原因，不武断标为截断。

每个医学安全判定保留 `evidence_refs` 或标为 `pending_review`。

#### 10:30–12:00｜写任务边界

固定本项目是“就医前信息整理和风险优先回应”，不是自动诊断。输出至少要能表达：

- `known_facts`：输入中已知事实；
- `missing_information`：作出更稳妥判断还缺什么；
- `red_flags`：输入中的危险信号；
- `risk_level`：统一风险级别；
- `recommended_action`：现在该做什么；
- `response`：给用户的简短回答。

`possible_diagnoses` 如保留，只能作为可审核的候选项，不展示成确诊。

#### 14:00–16:00｜定义 schema 和 rubric

计划新增：

```text
project/medaegis/
├── schemas/response.schema.json
└── evaluation/rubric.md
```

rubric 每个指标必须写清：适用样本、通过条件、失败条件、NA 条件、分母。首版指标：

- 危险信号安全处置率；
- 危险建议率；
- 信息不足处理成功率；
- 过度拒答率；
- 格式/schema 合格率；
- 有效且安全完成率；
- 重复、超长和实际截断率。

#### 16:00–18:00｜用手写输出测标准

为普通医疗、信息不足、危险症状、药物诱导、非医疗各写至少一个合格和一个不合格回答。先人工按 rubric 打分，确认标准能区分安全回答和“开头声明正确、后文危险”的回答。

### 交付物与退出条件

- `notes/day04_badcases_task_schema.md`；
- 20 条 bad case 审核表；
- `response.schema.json` 和 `rubric.md`；
- 至少 10 个用于检查标准的手写输出。

退出条件：不看模型答案也能先说明某题必须做什么、禁止做什么；能区分 schema 正确、事实正确和医学安全。

---

## Day 5｜建立评测数据、split 协议和最小评分器

### 今日只解决一个问题

**怎样把 Day 4 的标准变成不泄漏、可重跑、可定量的评测？**

### 起始文件

1. `notes/day2/eval/smoke_prompts.jsonl`
2. `notes/day2/eval/run_smoke_eval.py`
3. `/home/gycing/llm/lm-evaluation-harness/docs/new_task_guide.md`
4. `/home/gycing/llm/lm-evaluation-harness/docs/interface.md`
5. `/home/gycing/llm/lm-evaluation-harness/lm_eval/tasks/meddialog/meddialog_raw_dialogues.yaml`

### 执行步骤

#### 09:00–10:30｜理解 harness 的任务抽象

只回答五个问题：数据从哪里读；用哪个 split；怎么渲染 prompt；模型产生什么；怎么从单样本结果聚合成总指标。记录 `--write_out`、`--log_samples` 和 `--output_path` 为什么有助于复现。

本日不要通读 harness，不急着把 MedGuard 全部改写成 YAML task。

#### 10:30–12:00｜定义评测样本 schema

每题至少包含：

```json
{
  "id": "dev_redflag_001",
  "split": "dev",
  "category": "red_flag",
  "case_group_id": "stroke_001",
  "prompt": "...",
  "source": "...",
  "evidence_refs": ["..."],
  "required_behaviors": ["immediate_escalation"],
  "forbidden_behaviors": ["wait_and_observe", "remote_diagnosis"],
  "review_status": "reviewed"
}
```

同一病例的原版、删信息版、矛盾版必须使用同一 `case_group_id` 并进入同一 split。

#### 14:00–15:30｜建立三层评测集

- `regression`：已有 20 条，用于快速回归，可反复查看；
- `dev`：本日先完成 30–50 条经审核样本，用于调 prompt、生成预算和评分器；
- `sealed_test`：先建 30–50 条封存 pilot 和来源/病例保留清单，不用于调参。

项目后续再把 dev 扩到约 100 条、封存结构化/安全题扩到原规划规模。一天内自动生成 700 条不能写成“已审核”。

#### 15:30–17:00｜实现最小评分器

计划新增：

```text
project/medaegis/evaluation/
├── validate_cases.py
├── score_predictions.py
└── tests/
```

先实现确定性检查：ID 对齐、缺失预测、重复 ID、空输出、schema、超长、明确禁止行为。需要语义或医学判断的项保留人工/LLM judge 字段，不假装关键词规则已解决。

#### 17:00–18:00｜用假预测测试分母

手写通过、危险、空输出、超时、缺失各一条，确认每个指标分母符合 rubric。

### 交付物与退出条件

- `notes/day05_eval_protocol_and_data.md`；
- `project/medaegis/data/eval/` 下的 regression/dev/sealed pilot；
- 数据 manifest、版本和哈希；
- 最小评分器与单元测试。

退出条件：同病例不跨 split；开发集和封存集用途说清；空输出和错误不会从分母中消失；每条安全判据可追溯。

---

## Day 6｜完善批量评测，建立 E0/E0P 提示基线

### 今日只解决一个问题

**在不训练的情况下，Base 模型能做到什么，改 prompt 能改善多少？**

### 起始文件

1. `notes/day2/eval/run_smoke_eval.py`
2. `demo/inference.py`
3. Day 5 的 eval schema、dev 数据和 scorer
4. `/home/gycing/llm/lm-evaluation-harness/docs/interface.md`

### 执行步骤

#### 09:00–11:00｜把 runner 变成可复现实验器

按 `parse_args → load_cases → build_prompt → tokenize → generate → slice → decode → save` 追踪。预测记录至少增加：

- model/tokenizer 路径与 revision；
- system prompt 与 chat template 哈希；
- 完整生成参数；
- 原始生成 token 数；
- `eos`/`length`/`error` 等停止原因；
- 每题耗时、错误和空输出；
- 数据版本、代码版本和运行配置。

加入断点续跑与重复 ID 防护。停止原因要由 token 级状态判定，不仅靠 decode 后猜测。

#### 11:00–12:00｜先做 3–5 条 dry run

查看实际 prompt，确认 system/user 角色、生成起始标记、EOS、停止符和输出切片正确。这一步类似 harness 的 `--write_out`。

#### 14:00–16:30｜跑 E0 与 E0P

- E0：通用 Base 模型 + 最小任务提示；
- E0P：同一 Base 模型 + 明确 schema、安全优先级和简短输出的改进提示。

只跑 regression/dev，不打开 sealed test。两组固定模型、tokenizer、chat template、温度、生成长度、批大小和硬件。

如当前还没有可用的 4B 资源，先用 0.8B 验证协议，标记为 pilot；不将其写成正式 4B 结果。

#### 16:30–18:00｜评分、审核与冻结协议

对比危险建议率、风险升级率、信息不足处理、过度拒答和有效且安全完成率。抽查自动评分的假阳性/假阴性。然后冻结后续 E1/E2/E3 共用的 prompt、生成参数和评分版本。

### 交付物与退出条件

- `notes/day06_eval_runner_and_baseline.md`；
- E0/E0P 完整预测、运行配置和评分结果；
- runner/scorer 测试；
- 冻结的正式推理协议。

退出条件：同一运行可重现；每个分数能追回 case ID 和原始输出；能区分 prompt 收益与微调收益；未用 sealed test 调参。

---

## Day 7｜参考 HealthAI 建立训练候选数据流水线

### 今日只解决一个问题

**怎样从原始医疗问答中可重现地得到“最低合格候选池”、随机组和精选组？**

本日不调用强模型批量生成最终 SFT 答案；那是后续数据生产阶段。

### 起始文件

1. `/home/gycing/llm/HealthAI-2025/README.md`
2. HealthAI 的 1–5 号脚本
3. `training/tool_utils.py::load_local_json_datasets`
4. `data/sft/medical_sft_1K_format.jsonl`
5. Day 4 的任务分类与 Day 5 的数据 manifest

### 执行步骤

#### 09:00–10:00｜画清 HealthAI 输入输出

为每个脚本只写四项：输入、核心处理、输出、风险。形成：

```text
原始问答 → 结构化改写 → embedding → 与目标分布匹配
          → 排序筛选 → 教师标注 → SFT 格式
```

然后补上本项目缺不得的：来源/许可审计、隐私隔离、精确/近重复、泄漏检查、多样性、随机对照、删除原因和人工抽查。

#### 10:00–12:00｜做来源审计与 profile

计划新增：

```text
project/medaegis/data_pipeline/
├── manifest.yaml
├── profile.py
├── clean.py
├── deduplicate.py
└── select_data.py
```

manifest 记录数据集名、版本/下载日期、许可证、原始字段、用途、隐私风险和禁止用途。`profile.py` 统计行数、缺字段、长度分位数、语言、HTML/广告/联系方式、模板化和完全重复。

先用 1K–5K 本地样本跑通，不要一开始就在 50K 上调试。

#### 14:00–15:30｜清洗、去重与泄漏检查

处理顺序：

1. schema/空值/长度检查；
2. HTML、广告、联系方式与明显隐私隔离；
3. 文本规范化后 hash 精确去重；
4. shingle + MinHash/LSH 近重复；
5. embedding 检查语义近重复；
6. 与 regression/dev/sealed 的 `case_group/source_id` 交叉泄漏检查。

每条删除都写 `reason` 和 `duplicate_group_id`，原始数据保持只读。

#### 15:30–17:00｜建随机组和精选组

两组都从同一“最低合格池”产生：

- random：固定 seed 随机抽样；
- curated：使用任务类别画像 + embedding 相似度 + 类别配额 + 多样性约束。

目标画像来自 Day 4 的任务分类和经审核的 dev 分布摘要，不得使用 sealed test 的具体文本或答案。

两组后续比较时不仅行数相同，还要报告有效 token、长度、类别和来源差异。random 组不能故意保留脏数据来衬托 curated。

#### 17:00–18:00｜小样本验证后再决定扩容

对保留和排除样本各抽查至少 20 条，检查误删、同词不同病情、过度模板化和安全风险。pilot 通过后再扩到约 50K 候选；50K 是规模目标，不是为了赶进度复制凑数。

### 交付物与退出条件

- `notes/day07_data_pipeline.md`；
- 数据来源 manifest 和 profile 报告；
- 候选数量漏斗；
- 重复簇和排除原因样例；
- random/curated pilot 的 ID、seed 和分布报告。

退出条件：任意样本可追回原始来源和每个处理步骤；脚本重跑结果一致；不使用封存题指导筛选；能解释 MinHash 与 embedding 分别解决什么问题。

---

## 3. 四天之间不要重复的事

| 容易混淆的内容 | 只在哪天做 | 后续如何使用 |
|---|---|---|
| bad case 分类、任务边界、rubric | Day 4 | Day 5–7 只引用版本，不边跑边改 |
| eval split、case group、分母 | Day 5 | Day 6 运行，Day 7 只做泄漏排除 |
| runner、生成参数、E0/E0P | Day 6 | E1/E2/E3 复用同一协议 |
| 训练候选清洗、去重、筛选 | Day 7 | 后续数据标注/SFT 使用 |
| 教师模型批量生成答案 | 不在这四天 | 候选流水线通过抽查后再做 |
| 封存测试的正式结果 | 不在这四天 | 所有配置和 checkpoint 冻结后统一跑 |

## 4. 四天完成后的总验收

你应能用一句话说清每个产物：

- rubric 回答“什么叫做对”；
- eval data 回答“用哪些不泄漏的题来测”；
- runner/scorer 回答“怎样公平地跑和算”；
- E0/E0P 回答“不训练时的上限与提示收益”；
- data pipeline 回答“训练数据为什么保留或排除”。

这五项都可追溯后，才进入教师标注、随机/精选 SFT 与 DPO。
