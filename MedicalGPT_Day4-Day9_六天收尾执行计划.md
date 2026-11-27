# MedicalGPT：Day 4–Day 9 六天收尾执行计划（原十天格式修订版）

> 当前进度：Day 1–3 已完成。本文件沿用原十天计划的分时段操作、阅读顺序、交付物和退出条件，按新的数据与评测方案更新 Day 4–9。原十天文件保留供查阅。
> 六天主线：大规模 medical 清洗与 C-Eval 目标筛选 → 原问答 SFT 数据及共同安全样本 → SFT 对照 → 开发评测与偏好数据 → DPO及最终双轨评测 → 部署与复现。Day 9 合并原 Day 9–10，部分文档在训练等待时提前准备。
> 每日约 7–8 小时；数据生成和训练能否按时结束取决于实测吞吐、GPU 和教师预算。不能以省略审核或编造完成状态赶进度。
> 2026-11-10 更新：400条结构化病例+300条安全题、vLLM部署与完整性能矩阵恢复为正式必做项。第二seed仍为增强项。六天是目标排期，未完成上述必做项时顺延，不能降级为可选后宣称项目完成。
> 2026-11-11 更新：主数据源改为 `shibing624/medical` 中文 SFT；C-Eval 医学 `val` 用于目标向量筛选，`test` 封存评测；正式问答数据 E1/E2 各 20K，另加两组完全相同的 1K 结构化安全样本。已逐页参考 `medical_gpt学习参考.docx`，特别是新增第9–13页。下文是统一执行口径，旧规模和旧的“所有 C-Eval 均禁止做目标”规则由本版替代。

> 2026-11-23 修正：正式流程使用**全部约200万条中文原始训练数据**清洗去重形成的完整合格训练池；全池编码后进行目标向量筛选，不再先抽10万–30万条缩小正式检索范围。E1为同池未做目标向量筛选的随机20K，E2为全池向量筛选20K；两组共享S=1K。数据探索的详细操作在对话中协作推进；实测画像与后续清洗/训练完成状态分开记录。

## 如何使用这份文档

1. 每天按“今日目标 → 分时段操作 → 交付物 → 必须学会 → 退出条件”执行。
2. 目录以 `/home/gycing/llm/MedicalGPT` 为根；文中的新脚本是计划新增，不代表已存在。小脚本可合并实现，不必为每个步骤另写一套框架。
3. 日期采用学习日编号，不沿用原文的 9 月日历。完成本文件后不再重复执行其他扩展计划。
4. 原文中正确的知识点和操作原样保留；关键变更在相应位置直接写清，不需要对照多个文件猜优先级。
5. 对外项目名使用MedAegis，实现目录统一为`project/medaegis/`。规划中的后续脚本、数据和实验需以实际产物确认完成，不能仅凭计划宣称已完成。

| 学习日 | 当天主任务                                   | 看什么确认完成                           |
| ------ | -------------------------------------------- | ---------------------------------------- |
| Day 4  | medical画像、去重、C-Eval val检索与E1/E2候选 | 来源manifest、20K/20K清单、去污染报告    |
| Day 5  | QA格式转换、共同S=1K、评测协议               | 两组21K训练文件、共同S哈希、100条开发题  |
| Day 6  | 同一Base正式SFT                              | E1/E2 adapter、训练token与资源日志       |
| Day 7  | 模板诊断、知识/行为开发评测、DPO数据         | val与QA/行为开发三张表、偏好对、test配置 |
| Day 8  | DPO、三科test和700题最终评测                 | E0/E1/E2/E3知识表、五组行为表、bad case  |
| Day 9  | 部署、18配置性能矩阵、复现文档               | 可调用服务、逐请求性能日志、README和报告 |

## 数据来源与规模约定

- 正式原始候选：[`shibing624/medical`](https://huggingface.co/datasets/shibing624/medical) 的 `finetune/train_zh_0.json`。数据卡列出约195万条（1,949,972条），包含医疗对话、百科和知识图谱问答；以下载 revision 的实际统计为准。它是项目作者整理的公开数据，不代表每条回答经过医学认证。保存 revision、文件哈希、来源行号、许可及下载日期；不能为没有来源标签的记录凭空指定原科室。
- `instruction/input/output` 转为统一问题和回答；审核合格的原 `output` 用作问答 SFT 标签。保留原文和修改记录，不用模型无差别覆盖195万条回答，也不从回答补造患者事实。`valid_zh_0.json`、`test_zh_0.json` 保留原用途，先做跨集合去重。
- `pretrain/` 仅作 CPT 扩展的数据来源，`reward/` 的现成偏好标签仅作待审核参考；本期不自动启动 CPT。`FreedomIntelligence/Huatuo26M-Lite` 保留为历史来源/扩展，主实验不混入未记录的新数据源。
- HealthAI-2025：借鉴“语料找目标题、Top-k 平均相似度”的实现；Word 第12页的 Curation 2.0 是方法依据。其付费 GLM→R1 全量重写流程不再作为必要前置。
- SFT：E1/E2 从同一最低合格池选问答，使用相同审核、过滤、补样规则；另外共享同一份结构化安全训练集 S，教师只负责这部分的候选标签。两组不能一组用原回答、另一组全用教师答案。
- DPO：训练侧病例 + 开发集发现的错误类型；不把任何 C-Eval 题及开发/测试病例的改写放进训练。现有 `data/reward/dpo_zh_500.jsonl` 主要用于格式参考。
- 评测双轨：C-Eval 三个医学科目测知识；100条行为开发、400条结构化最终测试、300条安全最终测试测实际行为。已有20题仅作回归。医学判据附可追溯资料，疑难项标待审。

| 内容           | 本轮正式安排                                                                                 | 扩展保留                         |
| -------------- | -------------------------------------------------------------------------------------------- | -------------------------------- |
| 原始与候选池   | 全部约195万条画像、清洗、精确/近似去重；隔离保留集合后，全量合格训练候选编码与筛选；1K仅调试 | 调整分块和存储方式，不缩小正式池 |
| 问答 SFT       | E1随机、E2目标筛选，各 N=20K 合格样本，验证另计                                              | 各50K/100K的规模实验，独立编号   |
| 共同任务 SFT   | 两组各追加同一份 S=1K 结构化安全样本；正式每组共21K                                          | S的数量/混合比例消融             |
| DPO            | M=200–500 合格训练对，验证另计                                                              | 约 1K                            |
| 自建评测       | 100开发 + 400结构化最终测试 + 300安全最终测试 + 已有20回归                                   | 更多独立来源与场景               |
| C-Eval知识评测 | `basic_medicine`、`clinical_medicine`、`physician` 三科test全量；val仅开发使用         | CMMLU独立外部验证，第二seed      |
| 服务           | FastAPI + vLLM部署 + Transformers/vLLM完整性能对比                                           | Gradio可视化界面                 |

Day 4实测后冻结全量处理规则、N=20K、S=1K、M=200–500、主模型和预算。候选池大小由全量清洗、去重、来源隔离和去污染后的实际剩余量决定，不预设压缩到10万–30万。E1/E2来自同一个完整合格训练池C：E1按seed=42随机选20K，不使用目标相似度；E2对C中全部候选编码、评分后选20K。1K仅用于调试/吞吐估算，不替代正式全量流程。资源不足应分块、持久化、断点续跑并顺延，不以预算抽样悄悄改变实验。

“向量化之前 vs 之后”在本项目中具体指**未做目标向量筛选 vs 做了目标向量筛选**。embedding本身不会修改问答或SFT标签，真正改变数据分布的是排序与选择。两组均经相同清洗、去重、去污染与质量审核。主对照保持20K对20K；如另做完整清洁池全部训练，单列E_full，报告训练token、步数和成本，它不能单独隔离筛选收益。两组允许自然重合，不人为移除交集。

### C-Eval 各 split 的角色（先定用途，再做向量）

| 文件/集合             | 本项目用途                    | 允许的操作                                                                 |
| --------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| 三科dev               | 每科5条few-shot示例、接口调试 | 默认主评测0-shot；5-shot作预设补充，不能混算                               |
| 三科val               | 目标分布、筛选与评测协议开发  | 编码题干+四个选项；不编码答案/解析；成绩标“开发分数”                     |
| 三科test              | 最终知识测试                  | 固定后仅最终评测；去污染脚本可按预设规则排除训练重复，不输出检索排名供筛选 |
| medical训练validation | QA和结构化任务loss、PPL       | 与SFT训练及所有行为评测分开；E1/E2共用                                     |
| 自建行为dev/test      | 安全和结构化行为              | dev调试；test封存；不复制进训练                                            |

三科名称必须是 `basic_medicine`、`clinical_medicine`、`physician`，不是 `clinical`。不将兽医学混入人类医学主指标。官方仓库在2025-07-27公告开放完整test；实际使用时检查所固定revision的三个test标签均为A/B/C/D。旧缓存标签缺失应定位官方新版本，不能用val替代test还称独立评测。[C-Eval官方说明](https://github.com/hkust-nlp/ceval)、[数据文件与split](https://huggingface.co/datasets/ceval/ceval-exam)

去掉完全重复仍不能消除“用同一评测题选择训练分布”的偏差。因此本项目公开披露val参与筛选；最终test不参与相似度排名、调参、教师提示或checkpoint选择。基础模型曾接触公开基准的风险仍然存在，不能声称排除了全部预训练污染。

### Word 第9–13页如何落实到本计划

| 参考页 | 借鉴内容                              | 本项目的实施与修正                                                                                |
| ------ | ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 第9页  | medical大数据池、C-Eval三科、知识匹配 | Day 4大数据清洗与目标检索；PT/CPT保留扩展，避免只给某组换起点                                     |
| 第10页 | ShareGPT转换、LoRA推理与合并          | Day 5保留问答语义并转换；Day 6/8检查adapter与merged一致性                                         |
| 第11页 | SFT后分数下降、chat template排查      | Day 7先核对模板/选项评分/token边界，再检验遗忘假设；“小模型SFT必然退化”不是结论                 |
| 第12页 | text2vec、Curation 1.0/2.0、Top-5     | Day 4默认text2vec本地编码、k=5；N取20K而非照搬100K；相似度+多样性；保留检索方向诊断               |
| 第13页 | 去污染、loss/PPL、结果表、独立验证    | Day 4/7固定去污染与独立QA验证；Day 8报告test和自建700题。截图中的成绩、私有测试结论不是本项目结果 |

参考文档称为RAFT的“背景+原题+答案”构造不加入本项目的评测题训练；也不据此将一般的向量筛选称为RAFT。原文“对齐税/遗忘”的解释作为待检验假设，模板修正后仍下降也不能单凭一项分数证明原因。

效率安排：Day 4锁定评测来源和病例分组；Day 5–7在生成/训练等待时分批完成评测标签审核和服务骨架；Day 7在独立环境完成vLLM版本/模型兼容性预检；Day 8集中跑最终评测；Day 9完成服务后端对比与复现。提前准备不允许查看最终测试模型输出或把测试题用于训练。

E0=同一主模型的简单提示；E0P=同一主模型的明确任务提示；E1=random QA 20K + shared S 1K；E2=C-Eval-val-curated QA 20K + 相同S 1K；E3=E2之后安全DPO。行为评测中E0P/E1/E2/E3使用相同任务提示、模板和生成预算。知识评测统一使用选择题协议，E0/E0P是同一权重，不把行为提示的两种命名当两份独立知识模型。正式主模型沿用Qwen3.5-4B，先验证适配；0.8B调试结果单列。数据筛选收益仅由同一Base的E1/E2对照支持。

主流程：medical全部约200万条流式读取 → 全量规则清洗/规范化/精确去重/MinHash → 全量合格候选分块编码与语义近邻审核 → 按重复组隔离保留集合、固定规则去污染 → 冻结完整训练池C（不做预算抽样） → E1从C随机20K / E2对C全部候选按C-Eval val Top-5+多样性选20K → 各加相同S 1K → SFT → 开发错误分析 → DPO → C-Eval test与自建700题 → vLLM部署。

---

# Day 4｜数据审计、清洗、去重与目标筛选

## 今日目标

建立medical大规模数据流水线，冻结C-Eval三科的用途，得到同规模随机问答候选E1和目标筛选候选E2，以及共同安全样本S的训练侧保留来源。

开始前用 30 分钟复用现有 `notes/day2/e0_base/smoke_badcase_report.md`，定好“事实整理、缺失信息、风险优先、简短回答”的任务边界，并确定开发/测试来源保留规则。无需重新花一天整理已审核的20题。

## 09:00–09:45：阅读参考项目的数据构造思路

按顺序打开：

1. [HealthAI-2025 README](/home/gycing/llm/HealthAI-2025/README.md)
2. `1.批量推理构造数据集.py`
3. `2.向量化构造的数据集.py`
4. `2.向量化验证集.py`
5. `3.相似度匹配（矩阵计算法）.py`
6. `4.排序、筛选、批量推理.py`
7. `5.制作微调数据.py`

以上脚本均位于 `/home/gycing/llm/HealthAI-2025/`。先看每个文件的输入、输出和主流程，不直接执行其中的API调用。

同时打开`medical_gpt学习参考.docx`第9–13页，重点看第12页的Curation 2.0。原参考先GLM改写病例；本项目优先保留medical原问答，目标用C-Eval val而最终test隔离。标出原代码衔接问题：脚本1输出`huatuo18M_processed.jsonl`而脚本2读取`result.jsonl`；脚本4排序调用被注释、只生成Batch请求；脚本5转换示例被注释、结果为JSON数组。新流水线使用显式输入/输出参数。

画出参考项目流程，并在旁边写出本项目补充：

- 精确去重；
- MinHash/LSH；
- 语义去重；
- train/dev/test 泄漏检测；
- 类别多样性；
- 数据报告；
- 安全偏好；
- 消融实验。

## 09:45–10:30：学习清洗和去重

### 必须掌握

- 文本规范化：空白、全半角、异常字符；
- 精确哈希去重；
- shingle；
- Jaccard similarity；
- MinHash 近似 Jaccard；
- LSH 通过 bucket 减少比较次数；
- embedding 语义相似度；
- cosine similarity；
- 为什么 MinHash 和 embedding 不能完全互相替代。

### 筛选原则

1. 先便宜规则，后昂贵模型；
2. 先精确去重，后近似/语义去重；
3. 数据质量和数据多样性同时考虑；
4. 不能使用最终测试集指导训练数据筛选；
5. 每条删除规则都必须可统计、可解释。

## 10:30–12:00：数据探索

对本地`dataset/medical/finetune/train_zh_0.json`全量只读画像，中文valid/test保留原用途；记录revision、SHA-256、schema、实际条数、空值、长度分布、规范化精确重复、异常候选及固定种子随机样本。input为空不等于问题为空；prompt_text由非空instruction/input顺序拼接，output独立保留。

先用1K检查脚本，再扫描全部原始训练数据；1K仅用于调试和人工抽查，不缩小正式候选池。字符统计与token统计分开，正式token预算使用主模型tokenizer和训练模板。人工审核后形成清洗规则，画像结果不等于已生成清洁训练集。

脚本与结果统一位于`project/medaegis/`。本时段的逐步命令、报告解读和知识讲解在对话中协作完成，不写入计划文件。

数据流水线目录规划（目录树保留在计划中，详细操作在对话中讲解）：

```text
project/medaegis/data_pipeline/
├── 01_profile_raw_data.py          # 已实现：全量只读画像与随机抽样
├── 02_clean_and_deduplicate.py     # 已实现基础清洗与精确去重；MinHash/语义去重待实现
├── 02b_minhash_candidates.py       # 已编写：MinHash/磁盘LSH候选审核，测试及试跑待用户执行；不自动删除
├── 03_build_target_profile.py      # 待实现：C-Eval val目标与split清单
├── 04_embedding_select.py          # 待实现：全池向量评分与多样性筛选
├── 05_split_by_cluster.py          # 待实现：按重复组隔离保留集合
├── 06_check_benchmark_overlap.py   # 待实现：跨集合重复与基准污染检查
├── 07_build_sft_sets.py            # 待实现：构建E1/E2及共同S训练文件
├── 08_build_data_report.py         # 待实现：汇总数量漏斗与数据报告
└── config.yaml                    # 待实现：固定规则、模型版本和随机种子
```

编号用于组织文件，不代表严格执行顺序：分组隔离与固定规则去污染须在最终E1/E2清单冻结前完成。上述状态以2026-11-26核对结果为准，后续随实际实现更新。

## 14:00–15:30：清洗与去重

当前基础版为`basic_clean_exact_v6`，执行schema/空值检查、NFC与换行规范化、HTML结构转换、控制字符处理及全局精确去重。实体解码后继续转换显露的HTML，第二次结构转换不再次解码实体，以保护URL参数；多层转义或未知复杂结构整条隔离。HTMLParser支持段落、换行、简单列表、强调格式、链接和上下标；上下标转Unicode或显式记法。表格、图片、代码、嵌套列表、未知或歧义HTML保存原文并隔离整条问答。问题或回答在清洗前后包含U+FFFD时整条隔离，不声称检测了全部乱码。拒绝重复JSON字段；保留内部LF/TAB，将CRLF/CR/VT/FF/NEL统一为LF，移除其他C0/DEL/C1，不统一删除Unicode格式字符。候选仅保留清洗后非空flags；汇总保留前后计数。手机号、邮箱、URL保留并标记，尚未去标识化，不添加“驻站医”专项规则。清洁正文与去重比较键分开，同问题不同答案保留分组，后续不得跨集合。cleaned仍是中间候选池；MinHash、语义去重、隐私处理、内容质量与去污染尚未完成。始终从不可变原始数据重新运行。2026-11-26按用户本次授权代跑：12项测试通过，全量1,949,972条处理完成，保留1,909,144、重复40,780、隔离48；输出为project/medaegis/reports/day04_clean/train_full_v6/。实现状态与下列完整计划区分。

`02_clean_and_deduplicate.py` 处理顺序：

MinHash候选阶段拆为`02b_minhash_candidates.py`：依赖固定版本datasketch==2.0.0，默认前10K试跑，0表示全量。问题侧字符3-shingle、128维签名、16×8分带；LSH召回后计算问题/回答的真实Jaccard，输出候选及差异提示，暂不自动删除。索引跨批次全局持久化至SQLite；单次候选上限和截断查询单独统计，阈值及资源预算待试跑后冻结。该阶段尚未形成最终近似去重保留集，不得宣称全量近似去重已完成。

1. schema 检查；
2. 空值和极短样本；
3. HTML/广告/联系方式；
4. 明显隐私字段；
5. 精确哈希去重；
6. MinHash/LSH 近似去重；
7. 对全量清洗/词面去重后候选分块embedding编码，建立覆盖全池的近邻索引并复核语义近重复；缓存向量供后续筛选复用。先完成共同去重、保留集合隔离和去污染，再冻结E1/E2共同训练池C。不能机械误删同病名但年龄、否定、剂量、病程不同的病例；
8. 将淘汰原因写入 report，不要静默删除。

每条输出至少包含：

```json
{
  "source_id": "medical_revision_train_zh_0_line_000001",
  "source": "shibing624/medical",
  "instruction": "原instruction",
  "input": "原input，可空",
  "output": "原回答",
  "prompt_text": "仅问题侧字段拼接",
  "normalized_prompt": "保留否定、单位、数值的规范化问题",
  "prompt_hash": "问题哈希",
  "record_hash": "问答整体哈希",
  "quality_flags": [],
  "duplicate_group_id": "近重复病例组ID",
  "cluster_id": "主题多样性簇ID",
  "split": "train_pool"
}
```

### 筛选前先完成病例分组划分

先运行`05_split_by_cluster.py`，按duplicate_group_id隔离同病例/近重复，再划出共同训练池、共同任务样本S的来源、DPO来源、独立QA验证和结构化验证。主题cluster_id用于多样性，不能把所有同主题记录等同于同病例；如选择按主题簇整体划分，另外披露其分布变化。

已有medical valid/test先检查与训练库重叠；若用于QA loss/PPL，冻结具体文件、去污染清单和token mask。行为开发/测试病例独立保留。后续E1/E2只从同一训练池选择，不能从最终测试回收“好样本”。

## 15:30–16:30：建立C-Eval目标分布与隔离清单

`03_build_target_profile.py`加载`ceval/ceval-exam`的三个配置：`basic_medicine`、`clinical_medicine`、`physician`。冻结revision，统计每科dev/val/test实际题数、ID、标签有效性和哈希。subject+split+原id组成唯一ID。

导出两类文件：

- `data/targets/ceval_medical_val_queries.jsonl`：仅subject、id、question、A/B/C/D，不包含answer/explanation；供筛选程序读取。
- `data/eval/ceval/`：各科dev/val/test及manifest，保留评分标签；test仅最终评测和固定去污染检查可访问，不能输入筛选/教师流程。

向量文本固定为“科目+题干+全部四个选项”。候选侧只编码`prompt_text`，不加入output；不用GLM为题目补诊断或虚构病例。两侧格式不同会影响相似度，人工检查检索结果并记录局限。原来的30–50条任务画像保留为共同S样本的覆盖清单，不再是主问答筛选的目标向量。

读取本地`../lm-evaluation-harness/lm_eval/tasks/ceval/_default_ceval_yaml`：其默认`validation_split: val`。记入评测协议，避免Day 8误跑成val最终评测。

## 16:30–17:30：向量筛选与多样性

`04_embedding_select.py`：

1. 按Word第12页，首轮使用本地`shibing624/text2vec-base-chinese`作为可复现起点；记录revision、维度、池化、max_seq_length、batch size、截断率和归一化。它是通用中文句向量模型，医学相似度需要抽查。[模型卡](https://huggingface.co/shibing624/text2vec-base-chinese)
2. 编码C-Eval val查询及完整共同训练池C的全部prompt_text；候选向量优先复用上一步相同模型/revision/预处理配置的缓存。同一配置、L2归一化，分块保存向量与source_id索引，支持续跑。分块只是计算方式，所有合格候选都必须被评分；先对齐source_id，防止去重后行号错位。
3. 对每条候选计算它到目标题的Top-5余弦相似度均值，`k=min(5, 目标题数)`；即“语料找题”。使用分块矩阵乘法，不能建立195万×195万的候选自相似矩阵。候选内部去重使用近邻召回。
4. 保存Top-5目标ID、每项分数、均值；按均值排序，以稳定source_id打破同分。预选最多50K，再通过类别配额/聚类/MMR选出E2的20K；数量不足要明确报告。
5. 记录三科覆盖、问诊/百科/图谱任务比例、长度和重复组集中度。某科目标题多可能主导排名，必要时在开发阶段设置科目配额并冻结。
6. 从同一最低合格池固定seed=42随机抽取E1的20K；两组均排除相同的S/验证/评测保留来源，使用同一质量门槛和补样策略。记录重合率、token差异，不故意给随机组留脏数据。
7. 用50条候选做人工检索抽查；保留问题侧相关性与回答侧质量的独立标签，相似度高不等于回答正确。

检索方向诊断：可在同一小样本对比“目标题找语料”（Curation 1.0）与“语料找目标题”（2.0）的覆盖、重复率；两者不等价，不宣称2.0必然更好。正式固定2.0+多样性，额外训练消融另编号。E1/E2测的是整套筛选策略，不能把全部收益归给某个去重组件。

## 17:30–18:00：检查划分和泄漏

检查筛选前完成的分组划分，不能选完 N 条再按80/10/10拆成不足 N 条训练数据。验证样本另计。

`06_check_benchmark_overlap.py`检查训练/S/DPO与C-Eval所有题、自建validation/dev/test的来源ID、规范化哈希、shingle/Jaccard及语义近邻。val可以用于筛选分布，但题目本身及近重复仍不能进入训练。对test只允许按预设规则排除污染；输出排除ID和原因，不把test相似度传回排名器或生成器。高相似不自动等于泄漏，否定、数值、答案映射和病例身份需要复核。

先去污染再生成最终E1/E2清单；后发现污染则按原算法补样并重新冻结。test审核只做污染判定，不查看模型答案来调规则。公开基准无法排除基础模型历史训练污染。

## Day 4 交付物

- `notes/day04_data_pipeline.md`；
- 原始全量、清洗去重后全池、保留集合及完整训练池C的manifest；全池编码/评分覆盖率为100%，1K调试与正式运行分开统计；
- 规则清洗报告；
- 去重报告；
- C-Eval三科split manifest、val无答案目标向量及检索配置；
- E1随机20K、E2检索+多样性20K候选和共同S的来源清单；
- 无泄漏的 split；
- 数量漏斗、长度、类别和相似度图。

## Day 4 必须学会

- 能解释 MinHash/LSH；
- 能解释 embedding/cosine；
- 能说明为什么相似度高不等于质量高；
- 能从任意样本追溯到原始来源和处理步骤；
- 能检测数据泄漏。
- 能解释val定向筛选与test最终评测的边界，以及样本数、token预算和原始池规模的区别。

## Day 4 退出条件

- E1/E2 数量相同；
- 所有删除都有 reason；
- 同病例/近重复不跨split，高相似候选有复核记录；
- 抽查入选和淘汰样本各 30 条，结果基本合理；
- 能向面试官解释数据筛选算法和复杂度。
- C-Eval test未参与排序；两组来自同一完整训练池C；C中每条候选都有向量和筛选分数；数量与20K目标的差额明确，未完成则继续处理。

---

# Day 5｜问答格式转换、共同结构化安全数据与偏好准备

## 今日目标

把E1/E2各20K问答转换为可训练格式，保留审核合格的原回答；另外构造两组完全相同的S=1K结构化安全样本。冻结评测与Base开发基线；DPO今天只构造示例，正式偏好对在Day 7完成。

训练数据采用两种明确任务：普通医疗知识/问答使用自然语言回答；病例整理任务使用下文JSON Schema。分别设置QA与结构化任务提示；同一种任务在E1/E2训练和行为推理时保持一致，避免要求百科QA输出虚构的病例字段。模型不是所有场景都必须输出JSON，知识选择题使用独立统一的考试协议。

## 09:00–09:45：阅读仓库内的数据生成代码

按顺序点击：

1. `tools/convert_dataset.py`、`tools/validate_jsonl.py`
2. `training/supervised_finetuning.py`中的`preprocess_function`、角色解析和label mask
3. `role_play_data/README.md`、`role_play_data/llm_client.py`
4. `role_play_data/role_generate.py`及任意一个`roleplay_data_generate_*.py`

先完成`07_build_sft_sets.py`：将Day 4问答的prompt_text作为human，原output作为gpt。核对转换工具实际支持的字段，不能凭Word中的`convert_sharegpt.py`文件名假定本仓库有同名实现。随机抽查50条，检查instruction/input拼接、换行、字符串转义和回答是否对应。

重点学习方法，不直接使用角色扮演任务：

- provider 如何选择；
- API Key 如何从环境变量读取；
- 请求参数放在哪里；
- 如何解析响应；
- 如何处理并发、失败和重试；
- 如何保存中间结果。

## 09:45–10:45：确定输出 Schema

当天新建：

```text
project/medaegis/schemas/clinical_analysis.schema.json
project/medaegis/prompts/teacher_prompt_v1.md
project/medaegis/prompts/system_prompt_v1.md
project/medaegis/prompts/qa_system_prompt_v1.md
```

以下输入/输出Schema只用于共同结构化安全集S及病例服务，不用于把20K百科/图谱问答硬改成病例。S从训练侧保留的真实问诊输入及独立构造场景获取，覆盖危险信号、信息不足、矛盾、正常低风险、合理回答与合理拒答。

先固定输入病例 Schema：

```json
{
  "case_id": "case_000001",
  "case_text": "患者主诉、现病史、既往史和检查结果",
  "source": "shibing624/medical:finetune/train_zh_0.json",
  "category": "respiratory"
}
```

字段约束：

- `case_id`：全项目唯一、稳定，不因重新运行脚本而变化；
- `case_text`：去标识化后的病例文本；
- `source`：原始数据来源，用于数据血缘；
- `category`：用于分层统计和多样性控制，不作为模型答案泄露给模型。

输出字段固定为下列结构；`possible_diagnoses` 允许为空，不能在信息不足时强迫给诊断。保留 `triage` 命名，不再并用另一套 `risk_level`。增加 `response` 作为给用户的简短行动建议。

```json
{
  "possible_diagnoses": [
    {"name": "疾病名称", "certainty": "high|medium|low"}
  ],
  "evidence": ["来自输入的证据"],
  "missing_information": ["缺失信息"],
  "red_flags": ["危险信号"],
  "triage": "emergency|urgent|routine|self_care",
  "response": "基于已知信息的简短回应与行动建议",
  "safety_notice": "研究用途，不替代医生诊断"
}
```

### 为什么不生成长思维链

- 长推理过程不容易逐项验证；
- 强模型可能产生看似合理但虚假的推理；
- 项目真正需要的是可引用证据、缺失信息和安全行为；
- 面试时可以将其表述为“结构化证据摘要”，不要声称获得模型隐藏思维。

## 10:45–12:00：共同安全集先做50条试生成

当天新建：

```text
project/medaegis/data_generation/
├── generate_structured_sft.py
├── validate_structured_output.py
├── retry_failed_requests.py
└── build_sharegpt.py
```

从S的训练侧保留来源先生成50条，统计：

- API 成功率；
- 首次 JSON 解析率；
- schema 通过率；
- 平均输入/输出 token；
- 平均成本；
- 虚构证据；
- 漏掉明显危险信号；
- 输出过长；
- certainty/triage 非法枚举。

首次 JSON 通过率约95%可作为工程参考；还必须检查虚构证据、危险建议与风险遗漏。纯格式通过不能作为批量生产唯一依据。

## 14:00–15:30：完成问答质量检查，生成共同结构化安全集

E1/E2的20K问答分别完成相同的质量检查，审核合格的原output直接使用，不全量调用教师。发现错误或危险原回答进入quarantine；若重写，必须采用两组相同规则、模型、提示和审查方式，记录原文与重写比例，避免把“教师答案更好”混为“检索更好”。缺额按Day 4冻结规则补样。

教师批量处理的是共同S=1K：同一source_id只生成一次，合格数据以完全相同内容追加到两组。S不与20K问答、训练validation、行为dev/test或C-Eval题重叠。S的原回答可作为审核参考，但教师输入不能把原答案中的未提供事实补入病例。

执行策略：

- 输入只包含当前病例，不携带最终测试答案；
- temperature 使用较低值；
- 请求包含唯一 request_id；
- 原始响应、解析结果和错误分别保存；
- 每条最多自动重试两次；
- 仍失败的进入 quarantine；
- 支持断点续跑，不能重启后重复付费；
- API Key 只放环境变量。

记录字段：

```text
request_id
source_id
provider
model
prompt_version
created_at
input_tokens
output_tokens
estimated_cost
raw_response
parse_status
validation_errors
```

## 15:30–16:30：构造两类ShareGPT、混合并抽查

普通问答保留原回答：

```json
{
  "conversations": [
    {"from": "system", "value": "固定的医疗问答提示"},
    {"from": "human", "value": "instruction与input有序拼接的prompt_text"},
    {"from": "gpt", "value": "经审核的原output"}
  ]
}
```

共同S的结构化答案写入：

```json
{
  "conversations": [
    {"from": "system", "value": "安全与格式约束"},
    {"from": "human", "value": "病例"},
    {"from": "gpt", "value": "结构化JSON字符串"}
  ]
}
```

自动检查：

- conversation 顺序；
- 所有必要字段；
- triage/certainty 枚举；
- evidence 是否有输入原文支持；关键词命中只是初筛，还需检查否定、数值和时间是否被改变；
- token 长度；
- train/dev/test 文件不混合。

按QA与结构化S、随机与精选分层抽查共至少100条，给出pass/fail和失败类别，不只抽查看起来干净的答案。QA不检查triage字段；S检查全部Schema。疑难医学正确性标待审。

最终输出：

```text
data/sft/e1_random/train.jsonl        # 20K随机QA + 同一1K S
data/sft/e2_curated/train.jsonl       # 20K精选QA + 同一1K S
data/sft/shared_safety/train.jsonl    # S母版，hash一致
data/sft/validation/qa.jsonl         # 独立QA验证
data/sft/validation/structured.jsonl # 独立结构化验证
```

这里路径相对于`project/medaegis/`，均为计划产物。训练脚本可能读取目录下所有JSONL，S母版不要再复制到e1/e2训练目录旁导致重复加载。病例元数据、source_id和task_type可保留在旁路manifest，避免误当模型输入。统计实际有效训练token：不能只看21K行数，截断后的assistant标签也必须非空。

## 16:30–17:30：定义偏好类型并构造少量示例

当天新建：

```text
project/medaegis/preference/build_medical_preferences.py
project/medaegis/preference/validate_preference_pairs.py
```

先每类做少量示例，正式 M=200–500 对在 Day 7 根据开发错误构造。下表原1K配额保留为扩展参考，不是今天必须完成的数量：

| 类型       | 数量 | rejected 特征               |
| ---------- | ---: | --------------------------- |
| 过度确诊   |  200 | 信息不足仍给唯一高确定诊断  |
| 虚构依据   |  200 | 添加输入中不存在的症状/检查 |
| 红旗遗漏   |  200 | 危险病例仍建议居家观察      |
| 不安全用药 |  150 | 无依据提供具体处方/剂量     |
| 格式破坏   |  100 | 非法 JSON 或字段缺失        |
| 过度拒答   |  150 | 正常低风险问题也完全拒绝    |

chosen/rejected 需要：

- 回答同一 prompt；
- 长度尽量接近；
- chosen 确实更安全且更有用；
- rejected 的错误类型有标签；
- 不把明显错误医学知识写进 chosen。

### 利用批量生产等待时间：冻结评测并跑 Base 开发基线

复用 `notes/day2/eval/run_smoke_eval.py`，参考 `../lm-evaluation-harness/docs/new_task_guide.md` 和 `docs/interface.md` 的split、prompt、日志约定。补齐ID、配置、原始输出、生成token数、停止原因、错误、耗时；原日志缺停止原因写unknown。

基于Day 4保留来源，建设100条开发题、400条结构化最终测试题和300条安全最终测试题。两个最终集合互不重复，均与训练、验证和开发隔离。安全题固定组成：100条危险信号、80条信息不足、60条药物诱导、30条矛盾信息、30条非医疗/范围外。原20题继续只作回归。

每题包含case_group_id、来源、证据、required/forbidden behaviors和审核状态；同病例改写不跨split。Day 5先完成100条开发题和评测协议，最终700题分批审核可接续Day 6–7的训练等待时间，必须在Day 8评测前全部完成标签审核与版本冻结。记录合格、待审、排除的实际数量，不能把自动生成数量当合格数量。

先冻结题目、判据、分母和版本，再在开发集跑E0/E0P。用同一正式主模型；E0P/E1/E2/E3固定同一任务提示和生成参数。最终测试的模型输出此时不运行、不查看。审核耗时不够应记录缺口，不能把自动生成数写成合格题数。

## 17:30–18:00：数据卡和成本报告

写清楚：

- 数据来源与许可证；
- 清洗步骤；
- 强模型及提示词版本；
- 生成数量和失败率；
- 人工抽查方式；
- 不能视为临床金标准；
- 可能包含的偏差和风险。

## Day 5 交付物

- `notes/day05_data_generation.md`；
- JSON Schema；
- teacher/system prompt v1；
- 两组各21K混合SFT train、同一1K S母版、公共QA/结构化validation，以及独立行为dev/test；
- E0/E0P开发预测、评测manifest与固定推理协议；
- 少量DPO示例与偏好类型说明（正式数据Day 7完成）；
- quarantine；
- 成本与失败率报告；
- dataset card 草稿。

## Day 5 必须学会

- 会设计可验证输出协议；
- 会使用环境变量保护 API Key；
- 会实现重试、断点续跑和幂等；
- 会区分强模型生成标签与医学金标准；
- 会检查偏好对中的长度和风格偏差。

## Day 5 退出条件

- 两组ShareGPT训练schema通过率100%，S的输出schema通过；QA与S不同输出协议已验证；
- 首次生成 JSON 通过率达到约 95%；
- 抽查 100 条并完成失败分类；
- 偏好示例逐条核查，chosen 明确优于 rejected；
- 每条数据可追溯到来源和提示词版本。
- E1/E2问答各20K、共享S各1K，公共S哈希一致；实际token数、截断率、原回答保留/重写比例均已记录。

---

# Day 6｜正式 SFT、多卡训练与资源记录

## 今日目标

在完全相同配置下训练 E1（随机数据）和 E2（精选数据），获得可比较的模型。

## 09:00–10:00：复习训练参数与显存

点击：

1. `docs/training_params.md`
2. `scripts/run_sft.sh`
3. `scripts/zero1.json`
4. `scripts/zero2.json`
5. `scripts/zero3.json`

只理解 ZeRO 文件的核心含义，不在第一轮正式训练中使用。

必须能计算：

```text
global_batch = per_device_batch × GPU数量 × gradient_accumulation
```

必须知道显存组成：

- 模型参数；
- 梯度；
- optimizer states；
- activation；
- attention/KV 相关临时内存；
- CUDA context 和碎片。

静态显存估计只能作为起点，最终以 smoke test 实测为准。

在准备正式训练前，明确下载并校验主模型，而不是等 `torchrun` 启动后临时下载：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="Qwen/Qwen3.5-4B")
print(path)
PY

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

model_id = "Qwen/Qwen3.5-4B"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(config.model_type)
print(len(tokenizer), tokenizer.eos_token_id)
print(tokenizer.chat_template is not None)
PY
```

同时检查模型缓存目录和剩余磁盘空间。模型名、revision、缓存路径和文件哈希/commit revision 要写入 `experiment_manifest.json`。

## 10:00–11:00：准备可比较配置

当天新建：

```text
project/medaegis/configs/
├── sft_random.sh
├── sft_curated.sh
└── experiment_manifest.json
```

配置使用实际脚本支持的参数；当前不要假定训练脚本可直接读取任意YAML。两组使用相同训练设置、共同验证集和checkpoint选择规则，并匹配/报告有效token预算与长度分布。以下为待smoke验证的起点：

- Base：Qwen3.5-4B；
- 样本数：各N=20K QA，另加同一S=1K；每组21K，训练验证集另计；
- epoch：先短pilot估算耗时和过拟合，冻结两组相同预算；原3轮作为参考；
- model_max_length：1024；
- LoRA r=16、alpha=32、dropout=0.05；
- learning rate=`2e-5`；
- warmup ratio=0.03；
- seed=42；
- temperature 不属于训练参数，不要写入训练配置。

QA与S的混合比例、shuffle种子、task system prompt、assistant-only mask、截断规则必须相同；抽查两类解码结果。N相同不等于token预算相同，先报告监督token总量并尽量匹配长度；保留剩余差异，必要时增加等token预算对照。两组共同的loss验证目录可以含QA/S，但汇总loss之外需分别评估，避免两种任务的比例掩盖退化。

## 11:00–12:00：逐级 smoke test

依次验证：

1. 单卡20条、2–5 steps；
2. 只有准备实际使用多卡且获准时，验证对应GPU数的短运行；
3. 按实测步时、显存和可用时段估算完整E1/E2耗时后启动。

不强制逐级试遍两卡、四卡、八卡；DDP不使单卡模型显存自动均摊。

每一步检查：

- 数据数量；
- 解码后的 input/label；
- trainable parameters；
- loss 是否 NaN；
- 单步时间；
- 每卡显存；
- checkpoint 是否能保存。

## 14:00–14:45：启动 E1

E1使用随机20K问答+共同1K S，以当前获准资源执行保存好的命令：

```bash
bash project/medaegis/configs/sft_random.sh
```

保存到 `artifacts/e1_random_sft/`：config、command、logs、checkpoints、metrics、environment。

## 14:45–15:30：启动 E2 或准备串行接续

E2使用C-Eval val检索+多样性精选20K问答+同一1K S：

```bash
bash project/medaegis/configs/sft_curated.sh
```

有足够获准资源才并行，否则E1完成后串行；此时间块不意味着E1必定45分钟结束。保持模型、精度/量化、LoRA、优化器、全局batch、seed一致，记录实际训练token与更新步数。

训练等待时完成Day 7评分器和数据卡草稿，不重复阅读已掌握的Day 3源码。最终测试保持封存。

## 15:30–16:30：训练监控

使用：

```bash
nvidia-smi
tensorboard --logdir artifacts --port 6006
```

记录：

- step/time；
- train/eval loss；
- learning rate；
- grad norm；
- samples/s、tokens/s；
- GPU 利用率；
- 显存峰值；
- 异常重启次数。

### 不要用 loss 做的错误判断

- train loss 更低不一定表示测试更好；
- E1/E2 loss 不可直接说明数据质量；
- eval loss 使用不同数据时不可直接比较；
- 用相同开发集比较生成行为和选择checkpoint；最终冻结测试在Day 8统一运行。

Word第13页的loss/PPL只作为待复现实验形式。新增`evaluation/qa_nll.py`，在同一独立QA validation、相同tokenizer/template/截断/assistant mask上累加有效token的负对数似然：`PPL = exp(总NLL / 有效token总数)`。它是回答条件PPL，不是无条件医学语料PPL；不能简单平均每条样本PPL，也不能直接对不同任务/不同分母的Trainer loss取exp后比较。PPL下降不等于医学正确或安全性提高。

## 16:30–17:30：中间 checkpoint 检查

对 E1/E2 各加载一个 checkpoint，运行 Day 2 的 20 条 smoke prompts。

观察：

- JSON 是否更稳定；
- 是否出现模板复读；
- 是否过早停止；
- 是否把所有问题都当急症；
- 是否遗忘非医疗能力；
- 是否出现明显训练数据背诵。

## 17:30–18:00：无人值守准备

如果训练未完成：

- 确认日志持续写入；
- 确认 checkpoint 策略；
- 确认磁盘空间；
- 记录进程号和启动命令；
- 让训练自行运行，晚上不安排人工学习或值守。

## Day 6 交付物

- `notes/day06_sft_experiments.md`；
- E1/E2 配置；
- E1/E2 LoRA/checkpoint；
- TensorBoard 曲线；
- 显存、时间、吞吐报告；
- 固定样例中间对比。
- 公共QA验证的token加权NLL/PPL、监督token计数和两类任务分别的验证结果。

## Day 6 必须学会

- 会从单卡逐级扩到多卡；
- 会计算和保持全局 batch；
- 会区分训练 loss 与业务指标；
- 会保存完整实验上下文；
- 能处理 OOM、NaN、NCCL、磁盘满等常见错误。

## Day 6 退出条件

- E1/E2 使用相同超参数；
- 两组都生成有效 adapter；
- 日志和配置完整；
- adapter 可以重新加载并推理；
- 没有训练集/验证集重合。

---

# Day 7｜评测体系、SFT 对比和 DPO 数据定稿

## 今日目标

通过C-Eval val开发分数、独立QA验证和行为开发集检查筛选效果，排查模板影响，再用错误类型指导训练侧DPO数据；最终C-Eval test与自建700题留Day 8。

## 09:00–09:45：学习评测的基本分类

必须区分：

- closed-set：多项选择，适合 accuracy；
- open-ended：自由生成，需要解析和 rubric；
- format metric：JSON 合法率、字段完整率；
- factual grounding：证据是否来自输入；
- safety：危险建议、红旗召回、合理拒答；
- performance：延迟、吞吐、显存。

理解：

- exact match；
- precision/recall/F1；
- macro/micro average；
- Recall@k；
- bootstrap 置信区间；
- LLM-as-a-judge 的位置偏差、长度偏差和自偏好；
- benchmark contamination。

## 09:45–10:30：安装并认识 lm-evaluation-harness

优先在独立评测环境安装已导入的本地harness，记录commit与依赖版本；不要同时使用不明版本的pip发行版和本地源码：

```bash
python -m pip install -e '../lm-evaluation-harness[hf]'
lm-eval ls tasks | rg 'ceval|cmmlu'
```

先阅读`../lm-evaluation-harness/lm_eval/tasks/ceval/_default_ceval_yaml`以及三科的`ceval-valid_*.yaml`。本地配置为`validation_split: val`、`fewshot_split: dev`、`multiple_choice`、选项字符串A/B/C/D、指标acc/acc_norm。先在val用少量题验证（此处正式主模型已在Day 6验证可加载）：

```bash
lm-eval run --model hf \
  --model_args pretrained=Qwen/Qwen3.5-4B \
  --tasks ceval-valid_basic_medicine,ceval-valid_clinical_medicine,ceval-valid_physician \
  --device cuda:0 \
  --batch_size 1 \
  --num_fewshot 0 \
  --apply_chat_template \
  --limit 5 \
  --log_samples \
  --output_path artifacts/e0_base/ceval_val_smoke
```

实际 CLI 若与当前版本不一致，以 `lm-eval run -h` 为准，并把版本写入报告。

E1/E2分别将pretrained指向合并模型，或使用经验证的Base+adapter加载参数。命令只用于val开发，`--limit`结果不算完整分数。确认模型推理模式、chat template中的思考开关、候选token边界和special tokens；如需模板kwargs，先检查当前harness接口并记录配置。

### 参考Word第11页：先排查模板，再讨论遗忘

1. 在相同val题、相同权重上做raw prompt / 正确chat template对照，保存最终送入模型的文本和token；不可重复套两层模板。
2. 核对知识测试是否强行带入病例JSON提示、错误的assistant前缀、EOS或think配置。
3. `multiple_choice`比较候选A/B/C/D的条件log likelihood；它与“生成长回答后用正则提取字母”是不同协议。主指标固定前者，后者可单列诊断，不能挑较高分混报。
4. 在开发阶段冻结主协议（默认0-shot+适配主模型的chat template），所有模型同条件执行；5-shot从dev取例，只作为事先定义的补充。
5. 若正确协议下仍下降，结合公共QA PPL、任务比例、监督token、学习率、训练步数、知识/安全bad case进一步检验分布偏移或遗忘。不能把“修模板后仍低分”直接归因于容量或对齐税。

### 建立test专用任务，Day 8才运行

当天计划新增`project/medaegis/evaluation/ceval_tasks/`，配置与val分开。参考harness配置写`_default_medical_test_yaml`：

```yaml
dataset_path: ceval/ceval-exam
test_split: test
fewshot_split: dev
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案："
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: "{{['A', 'B', 'C', 'D'].index(answer)}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
```

每科建一个实际任务YAML，例如基础医学：

```yaml
include: _default_medical_test_yaml
task: medaegis_ceval_test_basic_medicine
dataset_name: basic_medicine
description: "以下是中国关于基础医学的单项选择题，请选出其中的正确答案。\n\n"
```

临床医学和医师资格同理修改task、dataset_name和description；manifest固定数据revision和三科数量。优先在任务配置中使用本地冻结快照，或按当前harness支持方式固定dataset revision并验证生效。Day 7只验证配置/人工玩具题的计分逻辑，不执行最终test模型预测。

报告`acc`为主、`acc_norm`为辅；每科单列。Micro=三科总正确数/总题数，Macro=三科准确率算术平均；都保存正确数与分母。多token字母和长度归一化可能影响两指标，不能默认永远相同。概率评分不需要temperature/max_new_tokens；这些参数属于另一张生成式安全评测表。

## 10:30–12:00：定义项目指标代码

当天新建：

```text
project/medaegis/evaluation/
├── generate_predictions.py
├── validate_json_outputs.py
├── clinical_metrics.py
├── safety_metrics.py
├── bootstrap_ci.py
├── qa_nll.py
├── ceval_tasks/
├── aggregate_ceval.py
└── error_taxonomy.py
```

### 使用 Day 5 已冻结的评测集，不重新造一套题

使用100条行为开发集做E0/E0P/E1/E2对比，400条结构化和300条安全最终测试保持封存。接续Day 5–6的标签审核，Day 7结束检查两套最终集合分别达到400/300条合格样本，完成病例隔离检查、manifest和哈希；未完成则Day 8先补审核，不用待审题凑数。

训练验证集用于loss；行为开发集用于提示、预算和checkpoint选择；最终测试只用于Day 8最终对比。三者不能混用。

必须执行以下隔离规则：

- C-Eval三科val可作为目标画像/相似度筛选，明确标为开发集；answer/explanation不进向量，原题和近重复不进训练。C-Eval test及可选CMMLU独立测试不进目标画像、筛选或教师生成；
- 自建最终测试集不得用于修改训练样本、提示词或 DPO 比例；
- 在第一次正式评测前冻结测试集；
- 为每个测试文件保存 SHA-256、样本数、类别分布和生成/审核版本；
- 后续如修正错误，必须增加新版本，不能静默覆盖旧测试集。

测试 manifest 至少记录：

```text
dataset_name
version
sha256
sample_count
category_counts
created_at
source
review_status
```

实现并写测试：

- JSON Valid Rate；
- Schema Completion Rate；
- Diagnosis Recall@3：仅有独立可靠诊断标签时启用，否则标未评测；
- Evidence Support Rate；
- Unsupported Claim Rate；
- Red-flag Recall；
- Triage Accuracy；
- Unsafe Advice Rate；
- Appropriate Abstention Rate；
- 过度拒答率、有效且安全完成率。

所有指标明确适用分母；错误/空输出不静默丢弃。安全评整段回答，不能只匹配关键词。无可核查断言时证据分NA，不能空回答得满分；医学疑难项标待审并报告覆盖率。

每个指标先手工构造 3–5 个例子，算出预期值，再写代码。

除指标测试外，必须额外补充：

- Schema 校验单元测试：合法输出、缺字段、非法枚举、非法 JSON；
- 数据划分测试：同一`duplicate_group_id`不得跨train/dev/test；主题cluster_id与病例重复组分开；
- 泄漏测试：确认的重复/近重复不得跨训练和test；高语义相似对进入复核，不以同病名直接判污染；
- 筛选隔离测试：目标ID只能属于三科val；目标文件无answer/explanation；test ID不得进入Top-k匹配记录；
- 混合数据测试：E1/E2共享S内容哈希一致，两组QA和S不重复计数；
- manifest 测试：文件内容变化后 SHA-256 必须变化。

## 14:00–15:30：运行知识、QA和行为开发评测

先跑完整三科val（去掉smoke的limit），输出E0/E1/E2的逐科acc、macro/micro；它已参与筛选，成绩只作开发分析。再在固定QA validation测NLL/PPL，最后运行100条行为dev。三张表分开，不能合成一个不明分母的“医学准确率”。

固定：

- 同一 dev manifest；
- E0P/E1/E2 同一 prompt version；E0 的提示不同是预设提示对照，单独标明；
- `temperature=0`；
- 同一 max_new_tokens；
- 同一 tokenizer/chat template；
- 同一后处理代码。

模型：

- E0：Base；
- E0P：Base + 固定任务提示；
- E1：随机20K QA + 共同1K S的SFT；
- E2：目标筛选20K QA + 同一1K S的SFT。

输出必须逐样本保存：

```json
{
  "sample_id": "dev_001",
  "model_id": "e2_curated_sft",
  "prompt_version": "v1",
  "raw_output": "...",
  "parsed_output": {},
  "parse_error": null,
  "metrics": {}
}
```

## 15:30–16:30：错误分析

给每个失败样本打一个或多个标签：

- `wrong_diagnosis`；
- `unsupported_evidence`；
- `missing_information_ignored`；
- `red_flag_missed`；
- `unsafe_medication`；
- `invalid_json`；
- `overconfident`；
- `over_refusal`；
- `too_verbose`；
- `truncated`：仅当原始生成状态支持时标记，否则unknown。

统计 E0/E1/E2 的主要失败类别，判断 E2 相比 E1 是否真的改善。

知识题另加`wrong_choice`、`option_scoring_mismatch`、`template_mismatch`，与行为错误分开。val/test结果不能混表；C-Eval相关训练样本只记录source_id和val匹配关系，不把test错题发给教师补训练。

## 16:30–17:30：据错误分析修订 DPO 数据

点击：

1. `docs/training_details.md` 中 DPO 部分；
2. `training/dpo_training.py`；
3. `scripts/run_dpo.sh`。

`training/dpo_training.py` 阅读顺序：

1. `ScriptArguments`；
2. `return_prompt_and_responses` 附近的数据 map；
3. 模型加载；
4. `LoraConfig`；
5. `DPOConfig`；
6. `DPOTrainer`；
7. `trainer.train`。

根据开发集真实错误类型调整偏好类别，不固定照抄Day 5配额。构造M=200–500对训练数据和另计验证对，题干来自训练侧保留病例，禁止将开发/测试题及改写放回训练。chosen需经审核，rejected必须确实较差；长度/风格匹配，抽查覆盖率和隔离原因写入数据卡。生成可从SFT等待时间开始准备。

## 17:30–18:00：DPO smoke 准备

验证：

- 每行包含 conversations/chosen/rejected；
- chosen/rejected 都是字符串；
- 两者回应相同 prompt；
- 没有 train/test 泄漏；
- 长度分布接近；
- 每个错误类型样本数可统计。

## Day 7 交付物

- `notes/day07_evaluation_and_dpo_data.md`；
- E0/E1/E2 指标表；
- 三科val开发结果、test专用配置、模板对照日志与公共QA验证NLL/PPL（正式test结果Day 8）；
- 逐样本预测；
- bad case 分类统计；
- DPO 数据最终版；
- 至少 6 个代表案例。

## Day 7 必须学会

- 能选择适合任务的指标；
- 能解释 LLM judge 的局限；
- 会保证模型间评测公平；
- 会从错误分析反推训练数据；
- 能读懂 DPOTrainer 的输入和配置。

## Day 7 退出条件

- E0/E1/E2 使用相同条件完成评测；
- 能说明开发侧是否支持“精选优于随机”，最终泛化结论留Day 8；
- 每个结论有原始预测支持；
- DPO 数据比例来源于真实错误，而非主观想象。

---

# Day 8｜DPO 训练、最终评测与消融

## 今日目标

训练 E3，验证安全性提升是否伴随可用性或知识能力下降。

## 09:00–10:00：理解 DPO

必须掌握：

- prompt、chosen、rejected；
- policy model；
- reference model；
- chosen/rejected log probability；
- 隐式 reward margin；
- beta 控制偏好强度与偏离参考模型的程度；
- DPO 不需要单独训练 reward model；
- DPO 不等于 PPO；
- 安全数据失衡可能导致过度拒答。

能用自己的话回答：

> SFT 教模型“应该怎样回答”，DPO 教模型“两个可能回答中更偏好哪一个”。

## 10:00–11:00：合并 E2 LoRA

点击：`tools/merge_peft_adapter.py`。

按顺序阅读：

1. 参数解析；
2. Base model 加载；
3. `PeftModel.from_pretrained`；
4. `merge_and_unload`；
5. tokenizer/config 文件复制；
6. `save_pretrained`。

计划命令：

```bash
python tools/merge_peft_adapter.py \
  --base_model Qwen/Qwen3.5-4B \
  --lora_model artifacts/e2_curated_sft/best \
  --output_dir artifacts/e2_curated_sft_merged
```

合并后先运行固定 20 条样本，确认与 adapter 加载方式结果基本一致。

## 11:00–12:00：DPO smoke test

配置建议：

- Base path：合并后的 E2；
- 50 对偏好数据；
- 单卡或双卡；
- 2–5 steps；
- LoRA r=16、alpha=32；
- learning rate=`5e-7`；
- beta=0.1作为参考：当前脚本未显式暴露/传入时，先确认并补通参数链路，打印最终DPOConfig，不假定命令参数生效；
- max source=768；
- max target=512。

检查：

- chosen/rejected 没有被截断为空；
- loss 是有限值；
- reward margin 方向合理；
- checkpoint 能保存和加载。

确认policy/reference从同一E2起点构造；`ref_model=None`不能直接理解为没有reference，应核对实际PEFT/TRL路径。Day 7若已完成数据与smoke，可提前启动DPO。未完成训练则顺延评测，不把时间表当作运行已完成。

## 14:00–15:30：正式 DPO

启动 E3，记录与 SFT 相同的实验信息：

- 配置；
- Git commit；
- 环境；
- 数据 manifest；
- GPU；
- wall-clock time；
- loss/reward margin；
- checkpoint。

不要只保存最终 adapter，还要保存用于选择 best checkpoint 的依据。

## 15:30–16:30：E3 安全检查

先运行20条回归、100条行为dev、三科C-Eval val和公共QA validation，按预设开发规则选择checkpoint与发布候选，然后冻结配置。随后对E0/E0P/E1/E2/E3统一运行400条结构化病例和300条安全最终测试；E0/E1/E2/E3统一运行C-Eval三科test全量。知识协议下E0和E0P是相同权重与同一考试提示，只计算一次；不在看完最终测试后反复选择模型。

五组模型共需3500条自建测试预测（外部任务另计）。用开发集实测的平均耗时估算总时长，支持按case ID断点续跑并保存错误；资源不足则顺延，不能按时钟删减模型或测试题。结构化和安全结果分表报告，分别保留适用分母。

### C-Eval正式test执行方法

Day 7已创建并验证test专用任务后，以E0为例：

```bash
lm-eval run --model hf \
  --model_args pretrained=Qwen/Qwen3.5-4B \
  --include_path project/medaegis/evaluation/ceval_tasks \
  --tasks medaegis_ceval_test_basic_medicine,medaegis_ceval_test_clinical_medicine,medaegis_ceval_test_physician \
  --num_fewshot 0 \
  --apply_chat_template \
  --device cuda:0 \
  --batch_size 1 \
  --log_samples \
  --output_path artifacts/e0_base/ceval_medical_test
```

其余模型使用已经验证的各自权重路径，协议不变；正式命令不带limit。实际chat template、思考模式、token边界、任务revision沿用冻结配置。执行前确认日志split=test、题数与manifest一致，不能因为任务名含test就跳过检查。API/harness失败要修复重跑或如实标缺失；不通过删失败题提高分数。

知识结果表：三科正确数/总数、acc、acc_norm、macro、micro、按科分层配对bootstrap差值区间。行为结果表：400结构化+300安全、各指标分母、审核覆盖率与病例组配对区间。另表记录QA验证PPL，PPL不能充当选择题准确率。

重点检查：

- 急症是否正确升级分诊；
- 信息不足时是否表达不确定性；
- 是否减少无依据用药；
- 是否出现所有问题都拒绝；
- 是否对普通非医疗问题仍有基本回答能力；
- JSON 是否仍稳定。

除了自动指标，随机抽取 50–100 条进行人工审查。审查时隐藏模型身份，使用统一 rubric 记录：

- 格式是否完整；
- 诊断是否与输入相符；
- 证据是否来自输入；
- 是否遗漏危险信号；
- 是否给出危险建议；
- 是否过度拒答；
- E2 与 E3 哪个回答更好，以及原因。

没有临床专家参与时，只能称为“人工代理审查”，不能写成“医生评测”或“专家评测”。

## 16:30–17:30：完整对比和消融

最终比较：

1. E0 vs E0P：提示改进是否有效；E0P vs E1：随机数据SFT是否有效；
2. E1 vs E2：相同20K QA规模、共同S和训练预算下，目标检索+多样性是否提高独立test知识分数，行为是否退化；
3. E2 vs E3：DPO 是否改善安全；
4. 可选：仅规则清洗 vs 规则+语义去重、Curation 1.0 vs 2.0、Top-5仅排序 vs 加多样性；若没有对应训练实验，只能报告检索数据诊断，不能声称组件训练收益。

补充区分：E0P→E1反映“QA+共同S的SFT整体收益”，不能声称单独证明了S或QA的作用；E1→E2才是筛选策略对照。若val提升而test无提升，报告可能的开发集适配/过拟合，不写成已证明泛化。若准确率下降，使用Day 7模板诊断、公共QA PPL和任务分布分析；不能直接照搬Word的遗忘结论。

报告：

- 点估计；
- 按病例组配对bootstrap 95% CI；单seed不能证明训练随机性稳定；
- 样本量；
- 评测脚本版本；
- 代表性成功/失败案例。

## 17:30–18:00：形成结论

只允许写数据支持的结论，例如：

- “目标筛选提升 JSON 合法率和证据支持率”；
- “DPO 降低危险建议，但轻微增加低风险病例的拒答”；
- “医学选择题分数没有明显提升，说明当前数据主要改善任务格式和安全，而非新增医学知识”。

不要写：

- “达到医生水平”；
- “显著提升”但没有置信区间；
- “医学准确率 99%”；
- 只挑成功案例。

### 预设效果目标（不是保证取得的结果）

以下原计划目标保留供对照，不能据此保证收益或在看到结果后修改口径。项目完成看实验链路和报告完整性，未达到效果目标仍可形成可信负结果。报告绝对失败数、分母和百分点差；基线危险建议率为0时相对下降不适用：

- E2/E3 的 JSON Valid Rate ≥98%；
- E2 的结构完整率和证据支持率应明确优于 E0/E1；
- E3 相比 E2 的 Unsafe Advice Rate 相对下降至少 30%；
- E3 的 Red-flag Recall 应高于 E2；
- E2相对E1的C-Eval三科test macro/micro差值及区间必须报告，无提升也保留；
- DPO后C-Eval三科test macro下降不超过2个百分点作为预设目标，同时报告micro和各科变化；不能用val升分代替test结果。

如果没有达到，也必须保留结果，并在报告中分析原因，不能删除失败实验或只展示成功样本。

### 已知限制清单

当天必须单独形成 `known_limitations.md`，至少包含：

- 强模型生成标签不等于临床金标准；
- 公开医疗问答自身可能包含错误和过时建议；
- 没有临床专家时，人工审核只能作为代理评测；
- 自建安全集规模和覆盖范围有限；
- 结构化指标改善不等于真实诊断能力提升；
- 4B 模型容量有限；
- 中文单一数据分布可能导致泛化不足；
- DPO 可能改善安全但增加过度拒答；
- benchmark 可能存在预训练污染；
- C-Eval val已用于目标筛选，开发分数不是独立泛化证据；去污染不能排除基础模型曾见过公开test；
- 20K QA加1K S是本轮混合比例，未做比例消融不能声称最优；
- 系统不能用于真实诊疗、处方或急救决策。

## Day 8 交付物

- `notes/day08_dpo_and_final_evaluation.md`；
- E3 adapter；
- E0–E3 完整指标表；
- 消融结果和置信区间；
- 安全收益与可用性损失分析；
- 最终 bad case 报告。
- 50–100 条人工代理审查记录；
- 五组模型各700条最终测试的预测或错误记录、两套测试manifest和逐类统计；
- E0/E1/E2/E3三科C-Eval test全量逐题概率/预测与acc、macro/micro、置信区间，开发/测试结果分表；
- `known_limitations.md`。

## Day 8 必须学会

- 能解释 DPO 目标；
- 会检查 reward margin；
- 会识别过度拒答；
- 会做公平消融；
- 会写不夸大的实验结论。

## Day 8 退出条件

- E3 可加载并推理；
- E0P/E1/E2/E3 全部完成同条件评测；E0作为提示对照单独报告；
- 400条结构化+300条安全题已完成统一评测，错误/超时保留，不按成功请求缩小分母；
- C-Eval三科test完成且split/题数/revision一致，没有用val替代；失败题和开发选择记录可追溯；
- DPO 安全收益和能力回退都有量化；
- 至少三项消融有原始结果；
- 预设效果目标已逐项判定，未达标项也被保留和解释；
- 完成人工检查和已知限制清单；
- 所有结论可追溯到数据和预测。

---

# Day 9｜模型部署、项目文档、复现与面试（合并原 Day 9–10）

## 今日目标

把实验模型变成可演示、可复现、能追问的项目成果，完成vLLM部署与Transformers/vLLM完整性能对比。数据卡、结果表、README草稿、API骨架和vLLM环境预检在前面训练等待时间准备。

当天时间分配：09:00–10:00模型与接口检查；10:00–11:00完善API；11:00–12:00启动vLLM并做正确性回归；14:00–15:00性能矩阵；15:00–16:00文档；16:00–17:00复现；17:00–18:00面试与验收。下文vLLM和性能操作详见“正式必做：vLLM部署与完整性能矩阵”。测试超过时段就接续，不缩减矩阵。

## 09:00–09:30：确定发布模型和安全边界

确认Day 8在开发侧已冻结的发布候选（E2或E3），依据不能只是“DPO更新”。最终test只报告效果、支持发布或阻止发布；若因test失败修改模型，旧test已成为反馈数据，需要新独立验证才能再次声称盲测。开发选择时考虑：

- 安全指标；
- JSON 合法率；
- 外部知识回退；
- 过度拒答；
- 推理稳定性。

模型卡必须写：

- 研究用途；
- 不替代医生；
- 不用于自动诊断或处方；
- 训练数据来源；
- 已知失败类型；
- 没有临床专家验证时明确说明。

## 09:30–10:00：点击推理和 API 文件

依次点击：

1. `demo/inference.py`
2. `demo/fastapi_server_demo.py`
3. `demo/openai_api.py`
4. `demo/gradio_demo.py`
5. `scripts/vllm_deployment.sh`

### `fastapi_server_demo.py` 阅读重点

- `Item` 请求模型；
- `/chat` 路由；
- 模型何时加载；
- 为什么 server 启动时只加载一次；
- 异常如何返回；
- CORS 是什么；
- 当前示例为什么还不满足医疗项目的结构化校验。

### 本项目接口

- `GET /health`：服务状态、模型版本；
- `POST /analyze`：病例分析；
- 输入必须限制长度；
- 输出先解析 JSON，再执行 schema 校验；
- 解析失败返回明确错误，不伪造结构；
- 每次返回免责声明。

## 10:00–11:00：完善本地推理 API

当天计划新建：

```text
project/medaegis/service/
├── app.py
├── schemas.py
├── model_backend.py
├── output_guard.py
└── system_prompt.py
```

先验证Transformers backend，再完成vLLM backend；二者使用同一发布模型和API输入/输出协议，分别保存原始生成结果及校验状态。

测试：

- 正常病例；
- 空字符串；
- 超长输入；
- 特殊字符；
- 非法模型 JSON；
- 服务内部异常；
- 并发两个请求。

## 15:00–16:00：完成 README、数据卡和模型卡

### README

README 固定结构：

1. 项目一句话介绍；
2. 问题定义；
3. 在MedicalGPT基础上实际完成的数据筛选、隔离评测与工程贡献；
4. 系统架构图；
5. 数据流水线；
6. 输出 Schema；
7. SFT/DPO 方法；
8. 实验矩阵；
9. 评测指标；
10. 结果表；
11. 消融实验；
12. 成功与失败案例；
13. 快速开始；
14. 复现命令；
15. API/Demo；
16. 数据和模型许可证；
17. 局限与医疗免责声明；
18. 致谢 MedicalGPT、HealthAI-2025、lm-evaluation-harness。

### README 最重要的图

- 数据数量漏斗；
- 系统架构；
- E0–E3 指标表；
- 数据筛选消融；
- DPO 安全收益/过度拒答对比；
- 两个成功案例和两个失败案例。

### 数据集卡和模型卡

数据集卡：

- 来源；
- 许可证；
- 数量；
- 清洗；
- split；
- 强模型标注；
- 人工抽查；
- 已知偏差。

模型卡：

- Base model；
- 训练阶段；
- 超参数；
- 数据版本；
- 评测；
- 使用方法；
- 限制；
- 禁止用途。

## 16:00–17:00：从零复现关键步骤与性能记录核对

不要在已有 shell 历史上复现。按 README 执行：

1. 新环境最小安装；
2. 100 条数据 smoke test；
3. 加载最终 adapter；
4. 运行 20 条评测；
5. 启动 API；
6. 发送请求。

记录所有缺失步骤，并立刻修订 README。

预热后固定输入和生成预算记录总延迟、实际输出token数、成功率与显存；没有流式时间戳时TTFT标未测。检查密钥、数据许可和权重是否误入Git，不把敏感值打印到报告。

### 简历和项目讲述（训练等待时准备，17:00后复核）

模板中的 `[X]` 必须来自真实结果：

> **MedAegis｜面向中文医疗问答与就医前信息整理的可靠性实验系统**
>
> - 基于 Qwen3.5-4B 与 MedicalGPT 搭建 SFT→DPO 训练链路，使用 `[实际GPU型号和数量]` 完成 LoRA 微调与可复现实验管理。
> - 基于shibing624/medical构建可追溯清洗流水线，在 `[实际处理数]` 条候选中按C-Eval医学val的Top-5相似度与多样性筛选 `[实际数量]` 条问答；与同规模随机SFT对照，独立C-Eval医学test的 `[macro/micro指标]` 变化 `[X]` 个百分点。
> - 构造 `[实际数量]` 组医疗安全偏好对进行 DPO，使 Unsafe Advice Rate 下降 `[X]%`、Red-flag Recall 提升 `[X]`，并量化过度拒答和知识能力回退。
> - 基于lm-evaluation-harness三科test与400条结构化/300条安全题完成Base、SFT、DPO对照，区分知识、安全与PPL指标，并通过 `[实际后端]` 部署，推理吞吐达到 `[X] tokens/s`。

不要写：

- 自主研发医疗大模型；
- 达到医生水平；
- 医学准确率 99%；
- 没有执行过的 PT/PPO/GRPO；
- 找不到原始实验文件的数字。

### 准备项目讲述

### 3 分钟版本

1. 20 秒：问题和目标；
2. 40 秒：数据为什么是难点；
3. 40 秒：SFT+DPO；
4. 50 秒：实验矩阵和结果；
5. 20 秒：部署；
6. 10 秒：局限和下一步。

### 10 分钟版本

增加：

- 数据去重和筛选算法；
- LoRA 与多卡配置；
- DPO 原理；
- 评测设计；
- bad case；
- 一次真实排障经历。

## 17:00–18:00：面试自测

必须能回答：

1. 为什么选 MedicalGPT，而不是 LLaMA-Factory？
2. 为什么选 4B，不直接训练 7B？
3. MinHash 和 embedding 去重有什么区别？
4. 相似度筛选是否会造成测试泄漏？
5. 为什么medical原回答可以审核后作SFT标签，却不能直接作为医学金标准？
6. 如何证明收益来自数据质量？
7. LoRA rank 越大越好吗？
8. SFT 的 label mask 怎么做？
9. 全局 batch 如何计算？
10. DDP 为什么每张卡仍有模型副本？
11. BF16 与 FP16 有什么差别？
12. QLoRA 的 NF4 是什么？
13. DPO beta 控制什么？
14. 为什么 DPO 可能导致过度拒答？
15. 为什么不做 PPO/GRPO？
16. LLM judge 有什么偏差？
17. JSON 输出失败怎么处理？
18. 如何检测数据泄漏？
19. 医疗系统最大的安全边界是什么？
20. 如果诊断指标没提升，这个项目还有什么价值？
21. C-Eval val用于检索后，为什么不能把val成绩称为独立测试成绩？
22. Curation 1.0与2.0检索方向有什么区别，Top-5平均分有什么偏差？
23. chat template修正、选项概率评分、真实知识遗忘怎样分别排查？
24. QA、结构化安全集S、DPO数据分别训练什么，两组为什么共享S？
25. PPL的token分母是什么，为什么不能比较不同验证集上的PPL？

答不上来的问题重新定位到代码或实验，不背标准答案。

在文档、复现和安全检查全部通过后，创建明确的 Git release/tag，而不是只在文字里写“最终版”：

```bash
git status --short
git tag -a medaegis-v1.0 -m "MedAegis reproducible v1.0"
git show medaegis-v1.0 --stat
```

创建 tag 前必须确认：

- 需要提交的代码和文档已经提交；
- API Key、模型权重、缓存和隐私数据没有进入 Git；
- README 中记录对应 commit；
- 实验配置、数据 manifest 和结果表能映射到该版本；
- 是否推送远程仓库由你之后单独决定，本规划不默认执行 `git push`。

## Day 9 交付物

- `notes/day09_deployment.md` 与 `notes/day10_documentation_and_interview.md`；
- 可加载模型、`/health`、`/analyze`、正常及异常请求记录；
- vLLM启动命令、独立环境版本、与Transformers的同输入正确性回归；
- 两后端×输入128/512/1024 tokens×并发1/2/4的完整性能矩阵、逐请求日志与汇总报告；
- README、dataset card、model card、实验报告与环境说明；
- 简历草稿、3分钟/10分钟讲述稿、面试问题答案。

## Day 9 必须学会

- 会合并和加载LoRA，能检查tokenizer、chat template和config；
- 会进行请求和输出校验，并定位错误；
- 能从头复现关键链路；
- 能区分实验事实、假设与局限，简历数字可追溯。

## Day 9 退出条件

- 新进程按README能加载模型、跑回归并调用API；
- E0P/E1/E2/E3都有真实记录，失败实验也保留；
- vLLM服务可调用，同模型/模板/参数回归通过或差异有解释；
- 两后端共18个性能配置完成测试，TTFT、延迟、吞吐、显存和失败率有原始记录；不支持的配置明确报告失败及原因，不默默省略；
- 没有秘密信息、隐私数据或大模型权重误提交；
- 能在3分钟讲清主线，能回答原20题及新增数据筛选/评测的5个面试问题。

---

## 正式必做：vLLM部署与完整性能矩阵

以下为Day 9正式任务，不是可选增强。独立环境兼容性检查提前到Day 7，最终模型的运行与测试在Day 9完成。Gradio界面仍为可选，不影响vLLM和性能矩阵的必做地位。

### 11:00–12:00：安装并启动 vLLM、正确性回归

vLLM 依赖可能与训练环境冲突。优先：

1. 阅读 vLLM 官方安装文档；
2. 记录训练环境 `pip freeze`；
3. 必要时新建 `medicalgpt-serve` 环境；
4. 使用 OpenAI-compatible server。

参考 `scripts/vllm_deployment.sh`，但修改：

- `--model` 指向最终 merged 模型；
- `--served-model-name medaegis`；
- 根据 4B 模型决定 TP=1 或 2；
- `--max-model-len`覆盖实际模板后输入+生成预算；性能矩阵含1024输入token时不能再把上下文上限设为1024，先验证2048等足够长度与显存；
- 不直接把服务暴露到公网。

启动后用同一发布模型、tokenizer/chat template、system prompt、生成预算和固定20条回归输入，对比Transformers与vLLM。检查停止符、截断、JSON解析、风险行为；不要求逐字一致，但明显偏差必须排查。Day 8模型能力表和后端性能报告分开，不把后端或服务规则收益计为训练收益。

### Gradio Demo（可选）

界面只需要：

- 病例文本输入框；
- 分析按钮；
- 结构化诊断候选；
- 证据；
- 缺失信息；
- 危险信号；
- 分诊等级；
- 永久可见免责声明。

不要添加：

- 自动开药；
- “诊断准确率”营销语；
- 模拟医生身份；
- 保存真实患者隐私数据。

### 14:00–15:00：完整性能评测（超过时段接续完成）

固定输入长度档位：

- 128 tokens；
- 512 tokens；
- 1024 tokens。

记录：

- Time to First Token；
- 总延迟；
- tokens/s；
- 并发 1/2/4 的吞吐；
- GPU 显存峰值；
- Transformers 与 vLLM 差异。

测试协议：

1. 两后端均使用相同最终merged模型、精度、GPU资源、模板和生成配置；记录版本与硬件。不能把量化或GPU数变化当纯后端收益。
2. 输入长度按应用chat template后的token数计算，固定128/512/1024三个档位；记录实际长度和构造方式。并发为1/2/4，共每后端9种、两后端18种配置。
3. 每个配置预热后测至少30个请求，记录实际输入/输出token数。固定max_new_tokens并保留EOS自然终止；输出长度不同必须披露，不能直接把短回答更快解释为加速。
4. 两后端均用流式客户端记录请求发出、首个非空内容到达、完成时间；报告TTFT和端到端延迟的p50/p95。流式API未测通时该项仍是待完成，不能用总耗时替代TTFT。
5. 同时报告成功率、错误/超时数、请求吞吐（完成请求/测量墙钟时间）、输出token吞吐（成功输出token总数/墙钟时间）、每请求生成速率及显存峰值。不要把每请求tokens/s简单相加当吞吐。
6. 记录Transformers后端的串行/排队/批处理策略及vLLM调度配置，明确比较的是部署方案。两个后端分别测，不让其同时占同一张卡互相干扰；保持外部负载尽量稳定。
7. 保存逐请求JSONL、配置、汇总CSV和图；OOM/超时如实记录。p95在少量请求上波动较大，结论标明样本量；没有原始测量不能填写数字。

### 服务测试和演示脚本

准备一个 3 分钟演示：

1. 展示项目问题；
2. 输入普通病例；
3. 展示结构化结果；
4. 输入急症病例；
5. 展示危险信号和分诊；
6. 展示 Base vs E3 对比；
7. 强调研究用途和限制。

# 附录：保留原计划的知识清单、目录和排错方法

## 2. 新手知识清单与学习深度

### A 级：必须会解释、会修改、会排错

#### Transformer/LLM

- token、embedding、attention、causal mask；
- next-token loss；
- logits、softmax、生成参数；
- KV Cache。

#### Tokenizer/模板

- encode/decode；
- BOS/EOS/PAD；
- chat template；
- padding/truncation；
- assistant-only loss mask。

#### 数据

- JSONL/ShareGPT/DPO；
- schema；
- 清洗、MinHash/LSH、embedding；
- split 和 leakage；
- 数据版本和血缘。

#### 训练

- SFT、LoRA、QLoRA；
- batch、梯度累积、学习率、epoch；
- BF16/FP16/NF4；
- gradient checkpointing；
- OOM、NaN、过拟合。

#### 分布式

- DDP；
- world size/local rank；
- NCCL；
- global batch；
- ZeRO-1/2/3 的区别。

#### 对齐

- chosen/rejected；
- DPO policy/reference/beta；
- 过度拒答；
- 灾难性遗忘。

#### 评测

- accuracy/F1/Recall@k；
- C-Eval val目标筛选、test隔离、0/5-shot、选项概率与chat template；
- 公共QA验证的token加权NLL、回答条件PPL及其局限；
- format、grounding、安全；
- bootstrap CI；
- 消融和控制变量；
- bad case。

#### 工程

- checkpoint/adapter/merge；
- TensorBoard；
- vLLM/FastAPI；
- 单元测试；
- API Key 和隐私。

### B 级：理解原理，但本期不执行

- PT/CPT训练与无条件语料PPL扩展（QA条件PPL已列为A级必做）；
- RM+PPO；
- GRPO；
- 自训练 tokenizer；
- mid-training；
- LongRoPE/LongBench；
- RAG；
- Data-Juicer；
- BERT 数据质量分类器。

### C 级：本期可以跳过

- Word2Vec 分层 softmax；
- Skip-Gram/CBOW 训练细节；
- 从零实现 Transformer；
- 自定义 CUDA/FlashAttention 内核；
- 7B 全参数训练；
- MoE 训练细节。

---

## 3. 统一实验目录建议

后续执行时，把自己的工作放在独立目录，避免与上游样例混合：

```text
project/medaegis/
├── configs/
├── data/
│   ├── raw_manifest/
│   ├── interim/
│   ├── targets/                 # C-Eval val无答案查询；无test筛选文件
│   ├── sft/
│   │   ├── e1_random/           # 20K QA + 同一1K S
│   │   ├── e2_curated/          # 20K QA + 同一1K S
│   │   ├── shared_safety/       # S母版，不被训练目录重复加载
│   │   └── validation/          # 公共QA与结构化验证
│   ├── preference/
│   ├── eval/
│   └── quarantine/
├── data_pipeline/
├── data_generation/
├── preference/
├── evaluation/                 # 含qa_nll.py、aggregate_ceval.py、ceval_tasks/
├── service/
├── prompts/
├── schemas/
└── reports/

artifacts/
├── e0_base/
├── e1_random_sft/
├── e2_curated_sft/
├── e2_curated_sft_merged/
└── e3_safety_dpo/

notes/
└── day01...day10
```

每个实验目录至少保存：

```text
config
command
git_commit
environment
data_manifest
logs
metrics
predictions
checkpoints/adapters
```

数据报告另存`raw_manifest.json`、`selection_config.json`、`selection_matches.jsonl`、`dedup_report.json`、`benchmark_overlap_report.json`、`split_manifest.json`、`data_report.json`。每条删除包含source_id、规则版本、原因、匹配保留ID；抽样裁剪单列，疑难项进quarantine。目标检索记录只能出现val ID。

知识预测至少保存subject、split、sample_id、model_id/revision、template_hash、各选项log likelihood、预测、参考标签、正确性、错误状态；行为预测另存原始输出、解析/审核标签、生成token、停止原因、耗时。Word成绩截图不进入本项目结果表。

---

## 4. 遇到问题时的排查顺序

### 模型下载失败

1. 检查网络；
2. 检查模型名；
3. 检查磁盘；
4. 检查 Hugging Face cache；
5. 不要重复下载到多个位置。

### CUDA OOM

1. 减小 per-device batch；
2. 减小 sequence length；
3. 开启 gradient checkpointing；
4. 增加 gradient accumulation 保持全局 batch；
5. 再考虑 QLoRA 或 ZeRO；
6. 不要第一步就改十个参数。

### Loss 为 NaN

1. 检查输入和 labels；
2. 检查是否全为 `-100`；
3. 检查学习率；
4. 检查 FP16/BF16；
5. 检查异常长样本；
6. 检查 grad norm。

### Loss 降但效果差

1. 检查训练/推理模板一致性；
2. 检查训练集质量；
3. 检查是否只学会格式；
4. 检查测试泄漏；
5. 检查生成参数；
6. 查看 bad case，不盲目增加 epoch。

### DPO 后全部拒答

1. 检查偏好数据类别比例；
2. 增加正常安全回答；
3. 检查 chosen/rejected 长度；
4. 先检查实际beta、学习率、步数与reference行为；按开发集一次只改变一个因素，不机械降低beta；
5. 对比 E2/E3 的正常问题可用性。

### 多卡 NCCL 报错

1. 单卡验证；
2. 两卡验证；
3. 检查 GPU 编号；
4. 检查端口冲突和残留进程；
5. 检查 NCCL 日志；
6. 退回已验证且获准的单卡/多卡配置，再定位通信问题。

---

## 5. 项目最终完成定义

只有同时满足以下条件，项目才算形成闭环：

- 数据从原始来源到训练 JSONL 可以重复生成；
- 原始全部约195万、清洗去重后的完整训练池C、全池编码/评分覆盖数、每组20K QA+共同1K S分别记账；清洗、保留集合隔离、随机对照抽取、检索和补样可追溯；不以10万–30万预算子集替代全池检索；
- E0、E0P、E1、E2、E3 全部有固定配置和结果；
- 至少完成随机 vs 精选、Base vs SFT、SFT vs DPO 三项消融；
- 关键指标有逐样本输出和置信区间；
- 成功和失败案例都被展示；
- Demo/API 能处理正常和异常输入；
- 400条结构化+300条安全最终测试已完成，保留全部预测/错误和审核记录；
- C-Eval医学三科test全量知识评测完成，val开发成绩与test成绩分开，附去污染记录和macro/micro；
- 公共QA验证PPL与模板诊断完成，未把低loss或参考截图当本项目医学效果；
- vLLM服务已部署并通过回归，Transformers/vLLM完整18配置性能报告可复核；
- README 能让另一个人在新环境跑通最小链路；
- 数据集卡、模型卡和医疗免责声明完整；
- 简历数字全部来自可定位的实验文件；
- 你本人能够解释数据、训练、评测、部署和失败原因。

这套标准比“把所有训练脚本都跑一遍”更接近实习面试官真正想看到的能力。
