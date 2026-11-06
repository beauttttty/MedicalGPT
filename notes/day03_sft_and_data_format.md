# Day 3：SFT 与数据格式

> 本页前半部分是结构化学习总结；文件末尾保留最初记录的原始数据样例。

## 1. PT/CPT、SFT、DPO 数据对照

| 文件                                     | 数据结构                              | 训练目标                                              |
| ---------------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| `data/pretrain/fever.txt`              | 普通连续文本                          | 根据左侧 token 预测下一个 token，学习医学文本分布     |
| `data/sft/medical_sft_1K_format.jsonl` | `conversations`                     | 给定问题，学习生成训练数据提供的 assistant 回答       |
| `data/reward/dpo_zh_500.jsonl`         | `conversations + chosen + rejected` | 对同一 prompt，相对提高 chosen 而不是 rejected 的偏好 |

`fever.txt` 用于已有模型上的医学领域训练时，更准确地称为继续预训练（CPT）。它没有角色和问答标签，通常令 `labels` 等于 `input_ids` 的副本，通过因果掩码逐 token 学习下一个 token。

SFT 样本中的：

```text
human → 用户问题，即条件/source
gpt   → 训练回答，即目标/target
```

应用 chat template 后，`input_ids` 同时包含问题和真实答案。默认 assistant-only loss 大致是：

```text
input_ids = [问题和角色标记][assistant答案][EOS]
labels    = [-100 ..........][assistant答案][EOS]
```

问题部分作为上下文被读取，但 `-100` 位置不计算 loss。普通 SFT 训练主要调用 `model(...)` 前向计算、`loss.backward()` 和优化器更新，不调用 `model.generate()`；后者主要用于推理、生成式评测或数据采样。

DPO 中：

```text
conversations → 两个候选回答共同面对的 prompt
chosen        → 数据标注认为更好的回答
rejected      → 数据标注认为较差的回答
```

`chosen` 只是偏好标签，不天然等于事实正确或医学安全。当前 `dpo_zh_500.jsonl` 中包含果汁口号、气候变化和电影标题等通用任务，适合学习格式和跑通流程，不能直接称为医疗安全偏好数据。

数据文件能被读取也不能证明质量合格：

```text
JSON语法正确
≠ 训练schema正确
≠ 任务相关
≠ 医学事实正确
≠ 回答行为安全
```

## 2. 长度偏差是什么

在 DPO 偏好数据中，长度偏差指 chosen 和 rejected 的长度存在系统性差异。例如：

```text
多数chosen：300～500 token
多数rejected：30～80 token
```

模型可能学到“越长越好”这一表面捷径，而不是我们真正希望它学习的医疗安全行为。反过来，如果 chosen 总是更短，也可能诱导模型把简短当作优质标准。

长度不同本身不一定错误。chosen 为了补充必要的急救提示而更长是合理的；目标是让两条回答的主题、格式和详细程度尽量接近，使主要差异集中在要训练的行为上。

较好的偏好对：

```text
chosen：识别严重过敏，立即建议寻求急救。
rejected：建议先观察1～2小时，未缓解再求助。
```

混杂因素过多的偏好对：

```text
chosen：一篇500 token、格式精美的医学文章。
rejected：不知道。
```

后一组同时混入了长度、格式、信息量和安全行为差异，无法证明 DPO 学到了哪一点。

优先比较 tokenizer 产生的 token 数，而不只比较汉字数：

```text
length_delta = chosen_tokens - rejected_tokens
length_ratio = chosen_tokens / max(rejected_tokens, 1)
```

至少报告 chosen/rejected 的均值、中位数、分位数、长度差方向占比和极端样本。SFT 也有长度分布问题：答案长期过长会使模型更啰嗦，超过最大序列长度还可能丢失答案尾部或 EOS。

## 3. JSON 和 JSONL 的区别

### 3.1 JSON

一个 JSON 文件整体表示一个顶层 JSON 值，常见的是数组：

```json
[
  {"id": 1, "prompt": "什么是高血压？"},
  {"id": 2, "prompt": "什么是糖尿病？"}
]
```

它可以跨行缩进，通常整体读取：

```python
with open("data.json", encoding="utf-8") as f:
    records = json.load(f)
```

### 3.2 JSONL

JSONL 是 JSON Lines，每个物理行是一条独立 JSON 记录：

```jsonl
{"id": 1, "prompt": "什么是高血压？"}
{"id": 2, "prompt": "什么是糖尿病？"}
```

它通常逐行读取：

```python
with open("data.jsonl", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
```

| 对比     | JSON                         | JSONL                             |
| -------- | ---------------------------- | --------------------------------- |
| 文件结构 | 一个完整顶层值               | 多条独立记录，每行一条            |
| 跨行缩进 | 可以                         | 一条记录不能跨物理行              |
| 读取     | 常用`json.load()` 整体解析 | 常用`json.loads(line)` 逐行解析 |
| 大文件   | 可能需要整体载入             | 可以流式处理                      |
| 追加     | 需维护整体数组语法           | 可以在末尾追加一行                |
| 报错定位 | 常是字符位置                 | 容易定位到具体行                  |

严格说，一条 JSONL 可以是任意合法 JSON 值；但本项目要求每行是对象，因为训练代码需要访问 `conversations`、`chosen` 等字段。

## 4. 为什么每一行必须是完整 JSON 对象

`tools/validate_jsonl.py` 的核心读取方式是：

```python
for line in file:
    data = json.loads(line)
```

程序每遇到一个物理换行，就把当前行单独送给 `json.loads()`。因此，下面是合法的普通 JSON 对象，却不是这个项目可接受的一条 JSONL 记录：

```json
{
  "id": 1,
  "prompt": "什么是高血压？"
}
```

校验器会分别尝试解析 `{`、`"id": 1,` 等不完整片段，所以失败。正确的 JSONL 是：

```jsonl
{"id": 1, "prompt": "什么是高血压？"}
```

如果字段内容需要换行，要在字符串中保存转义序列 `\n`：

```jsonl
{"answer": "第一段\n第二段"}
```

文件中仍是一条物理行；`json.loads()` 解析后，`\n` 才恢复成 Python 字符串中的真正换行。每行独立还便于流式处理、数据切分、追加记录和按行隔离错误。

## 5. `json.dumps(record, ensure_ascii=False)`

它把 Python 对象序列化为 JSON 字符串：

```python
record = {"content": "什么是高血压？"}
line = json.dumps(record, ensure_ascii=False)
```

转换关系：

```text
Python dict → json.dumps() → JSON str
```

`ensure_ascii=False` 让中文直接以可读形式保存，而不是强制变成 `\uXXXX`。它不会取消 JSON 必需的转义：字符串中的双引号、反斜杠、换行仍会写成 `\"`、`\\`、`\n`。

生成 JSONL 推荐使用：

```python
with open("data.jsonl", "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

不要手工拼接 JSON 字符串，否则遇到引号、反斜杠或换行很容易破坏格式。

四个常用函数：

| 函数                     | 作用                       |
| ------------------------ | -------------------------- |
| `json.dumps(obj)`      | Python对象→JSON字符串     |
| `json.loads(text)`     | JSON字符串→Python对象     |
| `json.dump(obj, file)` | Python对象直接写入JSON文件 |
| `json.load(file)`      | 从JSON文件读取Python对象   |

## 6. `tools/validate_jsonl.py` 检查了什么

使用示例：

```bash
python tools/validate_jsonl.py \
  --file_path data/sft/medical_sft_1K_format.jsonl
```

脚本逐行检查：

1. 当前行能否通过 `json.loads(line)`；
2. 是否包含 `conversations`；
3. `conversations` 是否为 list；
4. 每条消息是否包含 `from` 和 `value`；
5. `from` 是否属于 `system/human/gpt`；
6. 最后统计有效和无效行数。

此前实际检查结果为：

```text
medical_sft_1K_format.jsonl：1000/1000 行通过
dpo_zh_500.jsonl：500/500 行通过
```

这只能说明它们满足当前校验器覆盖的 JSON 语法和基础 conversations 结构。

## 7. 当前校验器没有检查什么

它没有完整检查：

- 每条 `conv` 是否真的是 dict；
- `value` 是否为非空字符串；
- system/human/gpt 的顺序是否合理；
- SFT 是否至少存在一个 gpt 回答；
- assistant labels 是否有效、是否全为 `-100`；
- DPO 是否包含 chosen 和 rejected；
- chosen/rejected 是否为非空字符串或是否完全相同；
- chosen 是否真的优于 rejected；
- chosen/rejected 是否存在长度偏差；
- 医学事实、证据和安全行为是否正确；
- 是否重复、泄漏到验证/测试集；
- 是否含隐私，来源和许可证是否允许使用。

例如下面可能通过当前基础校验，但没有可用的 SFT 回答目标：

```json
{"conversations":[{"from":"human","value":"什么是高血压？"}]}
```

下面可能通过 conversations 检查，却不是完整 DPO 数据：

```json
{"conversations":[{"from":"human","value":"问题"}],"chosen":"回答"}
```

另外，脚本提示文字写着角色应为 human 或 gpt，但实际代码也允许 system；提示文字与代码不完全一致。

## 8. 四层数据校验

```text
第1层：JSON语法
    └── 每行能否被 json.loads() 解析

第2层：任务schema
    └── 字段、类型、非空、角色顺序、chosen/rejected完整性

第3层：数据工程质量
    └── 去重、长度、截断、split泄漏、来源、许可证、隐私

第4层：语义与医学质量
    └── 相关性、事实、证据、安全性、偏好标签是否正确
```

当前 `validate_jsonl.py` 主要覆盖第1层和第2层的一小部分。程序不报错不等于训练数据正确；更危险的情况是程序顺利运行，却稳定地让模型学习错误行为。

## 9. 今日核心记忆

> JSON 是一个完整 JSON 值；JSONL 是逐行保存多条独立 JSON 记录。这个项目逐行执行 `json.loads(line)`，所以每一行必须独立、完整并满足任务 schema。格式校验通过只说明数据大致可读，不能证明训练目标正确，更不能证明医学内容可靠。

---

## 10. Causal LM 的训练目标：next-token prediction

### 10.1 Causal LM 学习的是什么

Causal Language Model（因果语言模型）的核心训练目标是：

> 给定当前位置之前的 token，预测下一个 token。

假设一句话经过 tokenizer 后是：

```text
<BOS>  我  爱  你  <EOS>
```

模型实际学习的预测关系是：

```text
看到 <BOS>       → 预测“我”
看到 <BOS> 我    → 预测“爱”
看到 <BOS> 我 爱 → 预测“你”
看到 ... 我 爱 你 → 预测 <EOS>
```

模型输出的 `logits` 形状通常是：

```text
[batch_size, seq_len, vocab_size]
```

含义分别是：

- `batch_size`：当前批次中的样本数量；
- `seq_len`：序列位置数量；
- `vocab_size`：每个位置对整个词表中所有 token 给出的原始分数。

`logits[b, t, v]` 表示：第 `b` 个样本的第 `t` 个位置，对词表中第 `v` 个 token 给出的预测分数。

### 10.2 Softmax 公式中的 i 和 j

对某一个序列位置，模型会将 logits 转换成词表概率：

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

其中：

- `i`：当前正在计算概率的那个词表 token；
- `j`：遍历词表中所有 token 的求和下标；
- `z_i`：模型给第 `i` 个 token 的 logit；
- `p_i`：第 `i` 个 token 的预测概率。

`i`、`j` 只是下标名字，可以替换成其他字母。为了避免与 batch 和序列位置混淆，更完整的语言模型记法常用：

- `b`：样本下标；
- `t`：序列位置；
- `v`：词表 token 下标。

### 10.3 Next-token cross entropy

如果正确的下一个 token 是 `y`，单个位置的交叉熵是：

\[
L=-\log p_y
\]

模型给正确 token 的概率越高，loss 越小；概率越低，loss 越大。

对多个有效 token 位置，一般再求平均：

\[
L=-\frac{1}{N}\sum_{n=1}^{N}\log p(y_n)
\]

其中 `N` 是参与 loss 计算的有效 token 数，不包括标签为 `-100` 的位置。

`CrossEntropyLoss` 直接接收原始 logits，内部已经包含与 `log_softmax + NLLLoss` 等价的计算，所以不要提前手动调用 `softmax`。

---

## 11. 为什么 labels 看起来没有右移

训练代码中经常有：

```python
labels = input_ids.clone()
```

因此表面上：

```text
input_ids：<BOS> 我 爱 你 <EOS>
labels：   <BOS> 我 爱 你 <EOS>
```

`labels` 本身没有提前移动。真正计算 loss 时，通过切片让 logits 和 labels 错开一个位置：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

### 11.1 `logits[:, :-1, :]`

三个维度分别代表：

```text
[batch, sequence, vocabulary]
```

所以：

```python
logits[:, :-1, :]
```

表示：

- 第一个 `:`：保留全部样本；
- `:-1`：保留第 0 位到倒数第二位，删除最后一个序列位置；
- 最后一个 `:`：保留完整词表维度。

最后一个 logits 位置没有序列内的“下一个 token”作为标签，所以被删除。

### 11.2 `labels[:, 1:]`

```python
labels[:, 1:]
```

表示保留全部样本，但从第 1 个序列位置开始，删除第 0 个 label。

第一个 token（例如 BOS）前面没有一个 logits 位置负责预测它，所以被删除。

### 11.3 具体对齐关系

```text
原序列位置：        0       1    2    3      4
token：           <BOS>     我   爱   你    <EOS>

保留的logits位置：   0       1    2    3
保留的labels：       我      爱   你   <EOS>
```

最终对齐：

| logits 位置 | 已看到的上下文     | 对应 label |
| ----------: | ------------------ | ---------- |
|           0 | `<BOS>`          | 我         |
|           1 | `<BOS> 我`       | 爱         |
|           2 | `<BOS> 我 爱`    | 你         |
|           3 | `<BOS> 我 爱 你` | `<EOS>`  |

所以更准确的说法不是“labels 张量被物理右移”，而是：

> 计算 loss 时删除 logits 最后一位和 labels 第一位，实现位置 `t` 的预测与位置 `t+1` 的正确 token 对齐。

Hugging Face 的 `AutoModelForCausalLM` 通常会在模型内部完成 shift。外部一般仍然传入形状相同的 `input_ids` 和 `labels`。

---

## 12. CrossEntropyLoss 代码语法和形状变化

常见实现：

```python
loss = CrossEntropyLoss(ignore_index=-100)(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1),
)
```

它等价于：

```python
loss_fn = CrossEntropyLoss(ignore_index=-100)

flat_logits = shift_logits.reshape(-1, vocab_size)
flat_labels = shift_labels.reshape(-1)

loss = loss_fn(flat_logits, flat_labels)
```

第一层括号：

```python
CrossEntropyLoss(ignore_index=-100)
```

用于创建并配置损失函数对象。

第二层括号：

```python
loss_fn(flat_logits, flat_labels)
```

用于调用损失函数。PyTorch 的损失函数继承自 `nn.Module`，对象可以像函数一样调用。

### 12.1 为什么要 reshape

假设：

```text
batch_size = 2
seq_len - 1 = 4
vocab_size = 10000
```

shift 后：

```text
shift_logits.shape = [2, 4, 10000]
shift_labels.shape = [2, 4]
```

执行：

```python
shift_logits.reshape(-1, vocab_size)
```

得到：

```text
[8, 10000]
```

表示 8 个 token 预测位置，每个位置都有 10000 个词表 logits。

执行：

```python
shift_labels.reshape(-1)
```

得到：

```text
[8]
```

表示这 8 个位置各自对应的正确 token ID。

`-1` 的含义是让 PyTorch 根据元素总数自动推断这一维，不是索引，也不是忽略标签。

默认 `reduction="mean"`，所以最终返回所有非 `-100` 位置 loss 的平均值。

---

## 13. 为什么 prompt token 通常设置为 -100

SFT 的输入通常同时包含：

```text
system prompt + user prompt + assistant answer
```

例如：

```text
<BOS> User: 什么是高血压？ Assistant: 高血压是…… <EOS>
```

如果只希望监督 assistant 回答，可以构造：

```text
input_ids：完整的prompt和answer token
labels：   -100 ... -100 + assistant答案token + EOS
```

简化示例：

```text
token：  <BOS> 用户 问题 助手 答案1 答案2 <EOS>
label：  -100 -100 -100 -100 答案1 答案2 <EOS>
```

`CrossEntropyLoss(ignore_index=-100)` 会忽略 label 为 `-100` 的位置。

这样做的目的包括：

1. 不要求模型学习复述 system/user prompt；
2. 把监督重点放在 assistant 应该如何回答；
3. 避免大量 prompt token 主导 loss；
4. 保留 prompt 作为回答生成的上下文。

### 13.1 -100 不等于删除 prompt

prompt token 仍然会：

- 输入 embedding 层；
- 经过各层 Transformer；
- 生成隐藏状态；
- 产生 attention 的 Key 和 Value；
- 被 assistant token 位置读取。

`-100` 只表示：

> 不把这个位置当作直接的预测目标计算交叉熵。

它不表示：

- 删除 prompt；
- 不计算 prompt 的隐藏状态；
- 阻止 assistant 关注 prompt；
- 切断整个 prompt 相关计算图。

### 13.2 为什么 prompt 仍会影响梯度

答案位置的隐藏状态依赖 prompt：

\[
h_{answer}=f(h_{prompt},\theta)
\]

答案 loss 又依赖答案隐藏状态：

\[
L=L(h_{answer})
\]

因此反向传播路径是：

```text
答案loss
  ↓
答案logits和hidden state
  ↓
assistant对prompt的attention
  ↓
prompt hidden states
  ↓
生成这些表示的可训练参数
```

prompt 没有直接 loss，但会通过答案 loss 间接影响梯度。

需要区分：

- `input_ids` 是离散整数，本身没有梯度；
- prompt 的浮点 embedding 和 hidden states 位于计算图中；
- 最终哪些参数更新，由 `requires_grad` 决定。

全参数训练时，相关 Base 参数可以更新；LoRA 训练时，梯度会穿过 Base 模型的计算路径，但通常只更新 adapter。

---

## 14. LoRA：用低秩矩阵表示权重更新

全参数微调直接更新完整矩阵：

\[
W=W_0+\Delta W
\]

LoRA 假设任务相关更新可以用低秩矩阵近似：

\[
\Delta W=BA
\]

前向传播写成：

\[
y=W_0x+\frac{\alpha}{r}BAx
\]

其中：

- `W0`：预训练 Base 权重，通常冻结；
- `A`：从输入维度降到 rank 维度；
- `B`：从 rank 维度升到输出维度；
- `r`：LoRA rank；
- `alpha/r`：LoRA 分支缩放系数。

假设原线性层权重形状为：

```text
W0.shape = [d_out, d_in]
```

LoRA 矩阵通常为：

```text
A.shape = [r, d_in]
B.shape = [d_out, r]
BA.shape = [d_out, d_in]
```

原矩阵参数量：

\[
d_{out}d_{in}
\]

LoRA 参数量：

\[
r(d_{in}+d_{out})
\]

当 `r` 远小于输入、输出维度时，可训练参数会显著减少。

### 14.1 Rank 决定适配容量

`rank` 控制 LoRA 更新矩阵的最大秩，也控制参数量和表达能力。

rank 较小：

- 参数少；
- 显存和 checkpoint 小；
- 正则化效果更强；
- 可能容量不足。

rank 较大：

- 表达能力增强；
- 参数、梯度和优化器状态增加；
- 可能更容易过拟合小数据；
- 不保证效果一定更好。

因此 rank 不是越大越好，需要用验证集比较。当前项目默认 `lora_rank=8`，运行脚本也设置为 8。

### 14.2 Alpha 控制 LoRA 缩放

标准 LoRA 常使用：

\[
\text{scale}=\frac{\alpha}{r}
\]

例如：

```text
r = 8
alpha = 16
scale = 2
```

alpha 控制 LoRA 分支对原模型输出的相对影响，但它与学习率、rank 和初始化共同作用，不能孤立理解为“越大越好”。

### 14.3 LoRA dropout

LoRA dropout 通常作用在进入 LoRA 分支的输入上：

```text
x → dropout → A → B
```

它只在训练时随机丢弃部分信息，用于正则化、降低过拟合风险；`model.eval()` 推理时 dropout 关闭。

dropout 太大可能损害学习，数据很少时也不能机械地认为 dropout 越大越安全。项目脚本使用 `lora_dropout=0.05`。

### 14.4 Target modules

`target_modules` 决定在哪些模块插入 LoRA，例如：

```text
q_proj, k_proj, v_proj, o_proj
gate_proj, up_proj, down_proj
```

不同选择会影响：

- 可训练参数量；
- 显存；
- 训练速度；
- 模型能够改变哪些计算路径。

项目中的 `target_modules=all` 不是训练所有参数，而是调用 `find_all_linear_names()` 找出符合条件的线性层，并排除 `lm_head` 和 `output_layer`，然后在这些线性层插入 LoRA。

### 14.5 Base 权重冻结，adapter 可训练

标准 LoRA 的预期状态是：

```text
Base weights：requires_grad=False
lora_A：requires_grad=True
lora_B：requires_grad=True
```

反向传播仍然经过完整模型，但优化器只更新可训练参数。这样可以减少：

- 参数梯度显存；
- 优化器状态显存；
- checkpoint 大小。

---

## 15. LoRA 的 A、B 初始化

原始 LoRA 采用：

```text
A：随机初始化
B：全0初始化
```

PEFT 默认通常使用：

```text
A：Kaiming-uniform随机初始化
B：全0初始化
```

两者的 A 随机分布细节不同，但因为 B 为 0：

\[
BA=0
\]

所以标准初始化时 LoRA 分支初始输出严格为 0，模型起点与 Base 模型一致。

### 15.1 为什么不能 A、B 都为 0

令：

\[
G=\frac{\partial L}{\partial(BA)}
\]

则：

\[
\frac{\partial L}{\partial B}=GA^T
\]

\[
\frac{\partial L}{\partial A}=B^TG
\]

如果 A、B 都为 0，那么两者梯度都为 0，LoRA 无法开始学习。

### 15.2 A 随机、B 为 0 时谁先更新

此时：

```text
dL/dB = G A^T，通常非0
dL/dA = B^T G = 0
```

所以第一次反向传播通常先更新 B；B 变为非零后，A 才开始获得有效梯度。

### 15.3 为什么不反过来使用 A=0、B随机

反过来也能够训练：

```text
A=0，B随机
```

此时：

```text
dL/dA = B^T G，通常非0
dL/dB = G A^T = 0
```

也就是第一步先更新 A，再更新 B。

两种初始化都能保证 `BA=0`，差异是第一步低秩更新方向不同。原论文选择 A 随机、B 为 0，主要因为：

1. A 从开始就能产生与输入相关的低维特征 `Ax`；
2. B 作为 LoRA 分支最后的输出投影，从 0 开始逐渐学习；
3. 符合残差分支“最后一层零初始化”的常见稳定设计；
4. 原论文和后续生态采用这一约定，方便复现。

因此 A=0、B 随机不是数学上错误，只是不是标准 LoRA 默认方案。

---

## 16. Gradient accumulation 与有效 batch

梯度累积把多个小批次的梯度累加后，再执行一次参数更新：

```text
micro-batch 1 → forward → backward → 累积梯度
micro-batch 2 → forward → backward → 累积梯度
...
micro-batch N → forward → backward → 累积梯度
                                   ↓
                            optimizer.step()
                                   ↓
                            optimizer.zero_grad()
```

有效样本 batch 的常用计算公式：

\[
\text{effective batch}
=\text{per-device batch}
\times\text{world size}
\times\text{gradient accumulation steps}
\]

项目 `run_sft.sh` 中：

```text
per_device_train_batch_size = 2
world_size = 2
gradient_accumulation_steps = 8
```

因此：

\[
2\times2\times8=32
\]

每张 GPU 每个 micro-batch 处理 2 条；两张 GPU 合计 4 条；累计 8 次后，才进行一次 optimizer update，相当于每次更新处理 32 条样本。

注意：32 是样本数，不是有效 token 数。不同样本长度不同，assistant 非 `-100` token 数也不同，所以每次更新的有效 token 数可能不同。

梯度累积主要解决单次大 batch 放不进显存的问题，但需要多次 forward/backward，通常不比真正的大 batch 更快。

---

## 17. Gradient checkpointing

Gradient checkpointing 更准确地说是 activation checkpointing，主要减少训练时保存的激活值。

普通训练：

```text
forward保存大量中间激活
backward直接使用这些激活
```

开启后：

```text
forward只保存少数检查点
backward需要时重新计算部分forward
```

本质上是：

> 用额外计算时间换取更低的激活显存。

它主要节省：

- hidden states；
- attention 中间张量；
- MLP 激活等。

它不直接减少：

- Base 权重；
- LoRA 权重；
- 优化器状态。

项目开启方式：

```bash
--gradient_checkpointing True
```

模型启用后还会设置：

```python
model.config.use_cache = False
```

因为用于自回归推理的 KV cache 在训练中通常不需要，并且通常与 gradient checkpointing 不兼容。

三个容易混淆的概念：

| 名称                     | 作用                       |
| ------------------------ | -------------------------- |
| gradient checkpointing   | 少保存激活，反向时重算     |
| gradient accumulation    | 多个小 batch 累积梯度      |
| model/adapter checkpoint | 把权重和训练状态保存到磁盘 |

---

## 18. QLoRA

普通 LoRA 通常是：

```text
Base：BF16/FP16加载，冻结
LoRA A/B：可训练
```

QLoRA 是：

```text
Base：4-bit量化加载，冻结
LoRA A/B：以更高精度训练
```

数据流：

```text
冻结的4-bit Base权重
       ↓ 计算时反量化到计算精度
参与前向和反向计算
       ↓
梯度穿过Base计算路径
       ↓
只更新LoRA adapter
```

QLoRA 不是直接训练 4-bit Base 权重。它的重点是降低冻结 Base 模型的权重显存。

QLoRA 论文的关键技术包括：

- NF4：适合近似正态分布模型权重的 4-bit 格式；
- Double Quantization：继续量化量化常数，进一步省显存；
- Paged Optimizers：缓解训练显存峰值。

项目代码中的典型配置：

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

根据当前项目逻辑，需要同时启用：

```bash
--load_in_4bit True
--qlora True
```

只设置 `--qlora True` 而没有加载 4-bit，不能形成真正的 QLoRA。当前 `scripts/run_sft.sh` 没有传这两个参数，所以默认执行普通 LoRA，而不是 QLoRA。

QLoRA 的主要优势是省显存、允许训练更大的模型，不代表训练一定更快；量化与反量化可能增加计算开销。

---

## 19. modules_to_save

标准 LoRA checkpoint 主要保存 `lora_A` 和 `lora_B`。但有时还需要完整训练并保存其他模块，例如：

- 新增的分类头 `classifier`；
- 新增 token 后的 `embed_tokens`；
- 需要完整训练的 `lm_head`；
- 自定义任务输出层。

PEFT 可以配置：

```python
modules_to_save=["classifier", "lm_head"]
```

其含义是：

> 除 LoRA adapter 外，这些指定模块也设置为可训练，并完整保存在最终 PEFT checkpoint 中。

需要区分：

| 配置                | 含义                           |
| ------------------- | ------------------------------ |
| `target_modules`  | 在指定模块中插入并训练低秩 A/B |
| `modules_to_save` | 完整训练并保存指定模块         |

`modules_to_save` 会增加：

- 可训练参数量；
- 梯度与优化器状态；
- checkpoint 大小。

如果 `embed_tokens` 和 `lm_head` 存在权重共享，还要检查模型的 weight tying，不能机械照搬配置。

---

## 20. 为什么要检查 requires_grad 和训练日志

配置名称表达的是“希望训练什么”，实际参数状态才说明“真正训练了什么”。

即使配置中有：

```text
use_peft=True
target_modules=all
```

也可能出现：

- 模块名称没有匹配；
- 某些 Base 参数意外解冻；
- `modules_to_save` 增加了完整可训练模块；
- 加载 adapter 时被设置为不可训练；
- `requires_grad=True`，但参数没有进入本次计算图；
- 参数没有进入 optimizer。

### 20.1 检查可训练参数名称

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape, param.numel())
```

标准 LoRA 通常主要看到：

```text
lora_A
lora_B
```

如果设置 `modules_to_save`，还可能看到：

```text
modules_to_save.default...
```

PEFT 也提供：

```python
model.print_trainable_parameters()
```

项目已经调用该函数打印 trainable params、all params 和 trainable percentage。

### 20.2 检查是否真的产生梯度

在一次 `loss.backward()` 后可以检查：

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            print(name, "grad=None")
        else:
            print(name, "grad_norm=", param.grad.norm().item())
```

这能区分：

- 参数允许训练但没有进入计算图；
- 参数实际获得了梯度。

LoRA 使用 A 随机、B 为 0 初始化时，第一次 backward 中 A 梯度可能为 0，而 B 有梯度，这本身不一定是错误。

---

## 21. Adapter checkpoint 与 safetensors

### 21.1 Adapter checkpoint 保存什么

LoRA 训练后常见输出：

```text
output_dir/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

其中：

- `adapter_config.json`：LoRA 的结构说明，例如 Base 模型、rank、alpha、target modules；
- `adapter_model.safetensors`：真正训练得到的 LoRA 张量数值，以及 `modules_to_save`；
- tokenizer 文件：训练脚本额外保存的分词器配置。

默认 adapter checkpoint 不包含完整 Base 模型，所以推理时通常仍需要：

```text
Base模型 + adapter checkpoint
```

### 21.2 Adapter checkpoint 与断点续训 checkpoint

仅有：

```text
adapter_model.safetensors
adapter_config.json
```

通常足够加载微调结果进行推理，但精确恢复训练还可能需要：

```text
optimizer.pt
scheduler.pt
trainer_state.json
rng_state.pth
```

因此：

- adapter checkpoint：保存模型增量，主要用于加载和推理；
- trainer checkpoint：还保存优化器、调度器、step 和随机状态，用于断点续训。

### 21.3 safetensors 是什么

`.safetensors` 是保存 Tensor 的二进制格式。它通常记录：

- 参数名称；
- shape；
- dtype；
- 数据在文件中的偏移位置；
- Tensor 的原始二进制数值。

逻辑结构可以理解为：

```text
Header长度
JSON Header：名称、dtype、shape、data_offsets
Tensor 1二进制数据
Tensor 2二进制数据
...
```

它不是源代码，也不是普通文本，用编辑器直接打开通常会看到乱码。与基于 pickle 的旧式文件相比，它只保存张量和有限元数据，不执行任意 Python 对象反序列化，安全性更好。

### 21.4 读取 adapter_model.safetensors

```python
from safetensors.torch import load_file

state = load_file(
    "outputs-sft-qwen-v1/adapter_model.safetensors",
    device="cpu",
)

for key in state:
    print(key, state[key].shape)
```

`state` 可以近似理解为：

```python
dict[str, torch.Tensor]
```

也就是：

```text
参数完整名称 → 参数Tensor
```

可能输出：

```text
...layers.0.self_attn.q_proj.lora_A.weight torch.Size([8, 1024])
...layers.0.self_attn.q_proj.lora_B.weight torch.Size([1024, 8])
```

其中：

```text
A.shape = [rank, in_features]
B.shape = [out_features, rank]
```

矩阵乘积 `BA` 与原线性层权重形状一致。

更完整的检查：

```python
from safetensors.torch import load_file

path = "outputs-sft-qwen-v1/adapter_model.safetensors"
state = load_file(path, device="cpu")

total_params = 0

for name, tensor in state.items():
    numel = tensor.numel()
    total_params += numel
    print(
        f"name={name}\n"
        f"  shape={tuple(tensor.shape)}\n"
        f"  dtype={tensor.dtype}\n"
        f"  numel={numel:,}\n"
    )

print(f"Total saved parameters: {total_params:,}")
```

通过键名和形状，可以验证：

- LoRA 是否插入预期层；
- rank 是否正确；
- 是否保存了 A/B；
- 是否意外保存了完整 Base 参数；
- `modules_to_save` 是否存在；
- checkpoint 的可训练参数规模是否合理。

---

## 22. LoRA 原论文应掌握的结论

阅读 LoRA 原论文时应重点关注：

1. 方法动机：全参数微调存储和部署成本高；
2. 核心假设：任务相关权重更新具有较低的内在秩；
3. 方法结构：冻结 `W0`，训练 `A、B`，用 `BA` 表示增量；
4. 初始化：A 随机、B 为 0，保证初始增量为 0；
5. 推理：LoRA 增量可以合并到 Base 权重中，避免额外推理延迟；
6. 实验结论：在论文测试的模型和任务上，以更少可训练参数达到与全参数微调相当或更好的效果。

论文中的具体数字只代表论文实验环境，不能直接当作当前 MedicalGPT 项目的实验结论。当前项目仍需要通过自己的：

- 验证集；
- 医疗安全评测；
- bad case；
- 显存与耗时记录；
- 不同 rank、数据和训练方法对照实验；

证明方法是否有效。

---

## 23. 本组知识的最终检查清单

学习完这一部分，应能够回答：

1. 为什么 Causal LM 的位置 `t` 预测位置 `t+1`？
2. 为什么要用 `logits[:, :-1, :]` 和 `labels[:, 1:]`？
3. `reshape(-1, vocab_size)` 中的 `-1` 是什么意思？
4. 为什么交叉熵输入是 logits，而不是手动 softmax 后的概率？
5. `labels=-100` 屏蔽了什么，没有屏蔽什么？
6. prompt 没有直接 loss，为什么仍然影响梯度？
7. LoRA 的 `A、B、rank、alpha、dropout` 分别做什么？
8. `target_modules` 和 `modules_to_save` 有什么区别？
9. 为什么 Base 权重冻结后，梯度仍能穿过 Base 的计算路径？
10. 为什么 LoRA 通常 A 随机、B 为 0？反过来是否可以？
11. gradient accumulation、gradient checkpointing 和 adapter checkpoint 有什么区别？
12. 为什么 `2 × 2 × 8 = 32` 只是有效样本 batch，不是有效 token 数？
13. QLoRA 与普通 LoRA 的 Base 权重精度和训练方式有什么区别？
14. 为什么不能只看配置名称，而要检查 `requires_grad`、梯度和 checkpoint 键名？
15. `adapter_model.safetensors` 和 `adapter_config.json` 分别保存什么？

核心总结：

> SFT 将完整对话作为条件输入，通过 next-token cross entropy 主要监督 assistant token；LoRA 冻结 Base 并用低秩 A/B 学习任务增量；gradient accumulation、activation checkpointing 和 QLoRA 分别从有效 batch、激活显存和权重显存三个角度解决训练资源问题；最终必须检查实际可训练参数、梯度和 checkpoint，而不能只相信配置名称。

---

## 24. `training/supervised_finetuning.py` 源码定点阅读

这一节把前面的通用原理落实到 MedicalGPT 的 SFT 脚本。不要逐行背诵，先掌握主流程：

```text
命令行
→ HfArgumentParser解析为4组参数对象
→ 加载tokenizer和特殊token
→ 加载/合并JSONL
→ conversations应用chat template
→ tokenizer编码并构造labels
→ 加载Base模型及权重
→ 可选量化和LoRA/QLoRA
→ DataCollator组装batch
→ Trainer调用模型forward并反向传播
→ 保存周期性checkpoint和最终结果
```

### 24.1 四组参数对象

脚本使用：

```python
parser = HfArgumentParser((
    ModelArguments,
    DataArguments,
    Seq2SeqTrainingArguments,
    ScriptArguments,
))
```

然后：

```python
model_args, data_args, training_args, script_args = (
    parser.parse_args_into_dataclasses(look_for_args_file=False)
)
```

命令行参数按照字段名进入对应数据类：

| 参数对象 | 负责内容 | 示例 |
|---|---|---|
| `ModelArguments` | 模型、tokenizer、精度、量化、缓存 | `model_name_or_path`、`load_in_4bit` |
| `DataArguments` | 数据路径和预处理 | `train_file_dir`、`max_train_samples` |
| `Seq2SeqTrainingArguments` | Trainer通用训练设置 | batch、epoch、lr、save策略 |
| `ScriptArguments` | 项目自定义SFT/PEFT设置 | `use_peft`、rank、模板 |

例如：

```bash
--model_name_or_path Qwen/Qwen3.5-0.8B
--train_file_dir ./data/sft
--learning_rate 2e-5
--lora_rank 8
```

分别进入：

```python
model_args.model_name_or_path
data_args.train_file_dir
training_args.learning_rate
script_args.lora_rank
```

`@dataclass` 是装饰器，它把普通类加工成数据类；“数据类”是加工后的类这一类别，二者不是同一个对象。

```python
field(
    default=None,
    metadata={"help": "..."},
)
```

中：

- `default` 是默认值；
- `metadata` 是字段附加信息；
- `help` 会被 `HfArgumentParser` 用于命令行帮助；
- `choices` 可限制允许值。

`__post_init__()` 会在数据类创建后自动执行，用于检查参数。例如脚本要求必须提供合法的 `model_name_or_path`，并要求 `model_max_length >= 60`。

### 24.2 常见参数含义

#### `model_name_or_path`

表示加载哪个模型，可以是：

```text
Qwen/Qwen3.5-0.8B              # Hugging Face模型ID
/home/gycing/models/qwen       # 本地模型目录
```

#### `cache_dir`

表示下载和数据转换缓存目录，不是训练结果目录：

```bash
--cache_dir ./cache
```

如果当前目录为项目根目录，则对应：

```text
/home/gycing/llm/MedicalGPT/cache
```

区别：

| 参数 | 含义 |
|---|---|
| `model_name_or_path` | 要加载哪个模型 |
| `cache_dir` | 下载/转换后的缓存放在哪里 |
| `output_dir` | 本次训练结果和checkpoint放在哪里 |

当前脚本将 `cache_dir` 传给了：

- `AutoTokenizer.from_pretrained()`；
- `AutoConfig.from_pretrained()`；
- Hub Dataset 的 `load_dataset()`；
- 本地 JSONL 的 Dataset/Arrow 缓存。

但当前 `model_kwargs` 没有放入 `cache_dir`，所以模型权重的 `AutoModelForCausalLM.from_pretrained()` 仍可能使用 Hugging Face 默认缓存：

```text
~/.cache/huggingface/hub
```

若希望模型权重也明确使用自定义缓存，应把：

```python
"cache_dir": model_args.cache_dir
```

加入 `model_kwargs`。

#### `rope_scaling`

用于尝试扩展 RoPE 模型的上下文长度，可设为 `linear` 或 `dynamic`。脚本根据目标 `model_max_length` 与模型原始最大长度计算缩放因子。没有超过原始长度时通常不需要开启，使用不当可能降低效果。

#### `max_train_samples` / `max_eval_samples`

用于快速调试和小规模实验。训练集先 shuffle，再 select 前 N 条；验证集则按代码截取前 N 条。`None` 或非正数通常表示不主动限制。

#### `glob`

```python
glob(f"{directory}/**/*.jsonl", recursive=True)
```

会递归搜索目录及所有子目录下的 JSONL，返回文件路径列表。

#### `num_proc`

```python
num_proc=data_args.preprocessing_num_workers
```

表示使用多少个 CPU 操作系统进程并行执行 Dataset 的 map/filter。它不是 GPU 数、torchrun 进程数、gradient accumulation，也不是 DataLoader worker。

### 24.3 本地 JSONL 的加载

如果设置了 `dataset_name`，脚本从 Hugging Face Dataset Hub 加载；否则搜索本地目录：

```python
train_data_files = glob(
    f"{train_file_dir}/**/*.jsonl",
    recursive=True,
)
```

当前 `data/sft` 下共有：

```text
medical_sft_1K_format.jsonl       1000条
sharegpt_zh_1K_format.jsonl       1000条
glaive_toolcall_zh_demo.jsonl      300条
```

因此 `--train_file_dir ./data/sft` 会加载全部三个文件，而不是只加载 medical 文件。

当前 `run_sft.sh` 同时设置：

```bash
--train_file_dir ./data/sft
--validation_file_dir ./data/sft
```

这意味着 train 和 validation 来源相同，存在明显数据泄漏风险。正式实验应使用互斥的训练集和验证集目录。

### 24.4 `load_local_json_datasets` 的职责

该函数位于 `training/tool_utils.py`，输入大致是：

```python
data_files = {
    "train": ["a.jsonl", "b.jsonl"],
    "validation": ["c.jsonl"],
}
```

执行步骤：

1. 创建空 `DatasetDict`；
2. 对每个 split 逐个调用 `load_dataset("json", ...)`；
3. 收集所有文件出现过的列；
4. 为缺失列补 `None`；
5. 统一列顺序；
6. 使用 `concatenate_datasets()` 拼接；
7. 返回 `DatasetDict`。

例如 medical 只有：

```python
{"conversations": [...]}
```

glaive 还有：

```python
{"conversations": [...], "tools": "[...]"}
```

合并后 medical 行会被补成：

```python
{
    "conversations": [...],
    "tools": None,
}
```

返回值形式：

```python
DatasetDict({
    "train": Dataset(...),
    "validation": Dataset(...),
})
```

访问单行：

```python
raw_datasets["train"][0]
```

返回一条样本字典。访问多行：

```python
raw_datasets["train"][:2]
```

返回按列组织的字典，例如：

```python
{
    "conversations": [第一条对话, 第二条对话],
    "tools": [None, None],
}
```

该函数只负责读取、字段对齐和拼接，不校验医学正确性、对话质量、重复或数据泄漏。

### 24.5 三类 SFT 数据的作用

#### Medical SFT

1000 条，全部是单轮 `human → gpt` 医疗问答，主要提供医疗领域问答形式和领域表达。但文件名为 medical 不代表回答正确或安全，仍需质量治理。

#### ShareGPT 中文数据

1000 条通用中文对话，其中大量为多轮，主要提供通用指令遵循、连续对话和上下文能力，不专属于医疗领域。

#### Glaive tool-call 数据

300 条，主要角色包括：

```text
human          用户提出任务
function_call  模型决定调用工具和参数
observation    外部工具执行结果
gpt            模型根据结果给最终回答
```

顶层 `tools` 是工具说明书，包含工具名称、描述和参数 schema。代码会按 `tool_format` 把它格式化并追加到 system prompt。

当前三个文件被混在同一个 SFT Dataset 中，并不会自动路由到不同模型层。

### 24.6 `preprocess_function(examples)` 的输入

因为：

```python
train_dataset.map(
    preprocess_function,
    batched=True,
)
```

所以 `examples` 是“一批样本组成的字典”，不是单条样本：

```python
examples = {
    "conversations": [
        [
            {"from": "human", "value": "问题1"},
            {"from": "gpt", "value": "回答1"},
        ],
        [
            {"from": "human", "value": "问题2"},
            {"from": "gpt", "value": "回答2"},
        ],
    ],
    "tools": [None, "工具JSON字符串"],
}
```

因此：

```python
examples["conversations"]
```

是多条 conversations 的列表；循环中的 `source` 才是一条样本的完整 conversations。

### 24.7 system prompt 从哪里来

脚本兼容两种格式。

方式一：conversations 内嵌 system：

```json
{
  "conversations": [
    {"from": "system", "value": "你是医疗助手"},
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "回答"}
  ]
}
```

方式二：顶层字段：

```json
{
  "system_prompt": "你是医疗助手",
  "conversations": [...]
}
```

对应：

```python
system_prompts = examples.get("system_prompt", "")
```

当前三个 `data/sft` 文件没有 system 角色，也没有顶层 `system_prompt`，所以普通样本中的：

```python
if role == "system":
```

通常不会成立。代码保留这些分支是为了兼容其他数据格式。Glaive 的 `tools` 内容则可能被追加到 system prompt。

### 24.8 从 conversations 到 dialog

代码先把角色统一：

```text
human/user/observation     → user侧
gpt/assistant/function_call → assistant侧
```

再整理为：

```python
history_messages = [
    [用户问题1, 助手回答1],
    [用户问题2, 助手回答2],
]
```

如果指定 `template_name`，使用项目 `training/template.py`；否则优先使用 tokenizer 自带的 `apply_chat_template()`。

最终 `dialog` 是 source/target 交替列表：

```python
dialog = [
    "<user>问题1<assistant>",
    "回答1",
    "<user>问题2<assistant>",
    "回答2",
]
```

所以：

```python
len(dialog) == 4
len(dialog) // 2 == 2  # 两轮问答
```

第 `i` 轮：

```python
source_text = dialog[2 * i]
target_text = dialog[2 * i + 1]
```

### 24.9 Tokenizer 编码与 labels

编码发生在 `Dataset.map(preprocess_function)` 内：

```python
source_ids = tokenizer.encode(
    dialog[2 * i],
    add_special_tokens=(i == 0),
)

target_ids = tokenizer.encode(
    dialog[2 * i + 1],
    add_special_tokens=False,
)
```

脚本按 source/target 原长度比例分配 `model_max_length`，截断后拼接：

```python
input_ids += source_ids + target_ids + [eos_token_id]
```

初始 attention mask：

```python
attention_mask = [1] * len(input_ids)
```

如果默认：

```python
train_on_inputs=False
```

则：

```python
labels += (
    [IGNORE_INDEX] * len(source_ids)
    + target_ids
    + [eos_token_id]
)
```

即：

| 内容 | 在 input_ids 中 | 直接参与 loss |
|---|---:|---:|
| system/user和角色模板 | 是 | 否 |
| assistant回答 | 是 | 是 |
| 回答后的EOS | 是 | 是 |
| batch padding | 是 | 否 |

多轮对话中每轮 assistant 回答都会参与 loss。全是 `IGNORE_INDEX` 的样本会被过滤。

### 24.10 `IGNORE_INDEX`、PAD 和 EOS

```python
IGNORE_INDEX = (
    LabelSmoother.ignore_index
    if data_args.ignore_pad_token_for_loss
    else tokenizer.pad_token_id
)
```

这是条件表达式。默认 `ignore_pad_token_for_loss=True`，因此：

```python
IGNORE_INDEX = LabelSmoother.ignore_index = -100
```

区别：

| 值 | 含义 |
|---|---|
| `-100` | loss忽略标记，不属于词表 |
| `pad_token_id` | 词表中真实padding token的整数ID |
| `eos_token_id` | 词表中EOS token的整数ID |

Tokenizer 同时有字符串和数字形式：

```python
tokenizer.eos_token       # 例如 "</s>"
tokenizer.eos_token_id    # 例如 248044
```

在训练文本 encode 之前，Tokenizer 对象已经通过 `from_pretrained()` 加载了词表，所以已经能知道特殊 token 的 ID。若缺少 EOS/PAD，脚本会注册或复用现有 token。

EOS 参与 labels，使模型学习回答结束后停止；PAD 的 label 使用 `-100`，避免 padding 参与 loss。

### 24.11 AutoModel 如何选择模型结构并加载权重

项目调用：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    **model_kwargs,
)
```

完整关系：

| 内容 | 保存位置 | 作用 |
|---|---|---|
| 结构代码 | `modeling_qwen3_5.py` | 定义层、Attention、MLP和forward |
| 结构参数 | `config.json` | 层数、维度、头数、词表大小 |
| 权重数值 | `.safetensors` | 训练得到的矩阵数值 |

Qwen3.5-0.8B 的实际 `config.json` 包含类似：

```text
model_type = qwen3_5
hidden_size = 1024
num_hidden_layers = 24
num_attention_heads = 8
num_key_value_heads = 2
intermediate_size = 3584
vocab_size = 248320
eos_token_id = 248044
```

`from_pretrained()` 大致过程：

```text
解析本地目录或HF模型ID
→ 读取config.json并创建Config对象
→ 根据AutoModel映射选择Qwen3_5ForCausalLM等具体类
→ 调用具体类__init__(config)创建Embedding、层和lm_head
→ 查找model.safetensors或分片index
→ 从safetensors读取Tensor
→ 按参数名和shape装入模型
→ 根据dtype/device_map/量化配置放到设备
→ 返回可运行的model对象
```

模型对象加载完成后，每次 forward 不再重新读取权重文件，而是使用 CPU 内存或 GPU 显存中的 Parameter。

`model.safetensors.index.json` 保存：

```text
参数名称 → 位于哪个权重分片
```

例如：

```text
model.language_model.layers.23.self_attn.q_proj.weight
→ model.safetensors-00001-of-00001.safetensors
```

### 24.12 不同模型的 forward 在哪里

`AutoModelForCausalLM` 是工厂，不要求 MedicalGPT 重写 `forward()`。它根据 Config 映射返回具体类：

```text
Qwen3_5Config → Qwen3_5ForCausalLM
LlamaConfig   → LlamaForCausalLM
MistralConfig → MistralForCausalLM
```

具体模型的 `modeling_xxx.py` 已经实现 `forward()`。Qwen3.5 Causal LM 的逻辑可简化为：

```python
outputs = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    ...
)

hidden_states = outputs.last_hidden_state
logits = self.lm_head(hidden_states)

if labels is not None:
    loss = self.loss_function(logits, labels, ...)
```

推荐调用：

```python
model(**batch)
```

而不是直接 `model.forward()`，因为前者先经过 `nn.Module.__call__()`，能够正确执行 hooks、autocast、PEFT 和分布式包装。

在本项目中，显式的 `model(**batch)` 被封装在 Hugging Face Trainer 内部；`trainer.train()` 会调用模型 forward、读取 loss、backward 并执行优化器更新。

### 24.13 量化、LoRA 和 QLoRA 的判断

三个关键开关控制不同层面：

| 参数 | 控制内容 |
|---|---|
| `load_in_4bit/load_in_8bit` | Base权重如何加载 |
| `qlora` | 4-bit时是否使用NF4+double quant配置 |
| `use_peft` | 是否插入LoRA/PEFT adapter |

典型组合：

| 配置 | 结果 |
|---|---|
| `use_peft=True, load_in_4bit=False` | 普通LoRA |
| `use_peft=True, load_in_4bit=True, qlora=True` | QLoRA |
| `use_peft=True, load_in_4bit=True, qlora=False` | 通用4-bit Base + LoRA，未启用项目完整QLoRA配方 |
| `use_peft=False, qlora=True` | 配置矛盾，没有adapter就不是QLoRA |
| `use_peft=False, load_in_4bit=True` | 试图全参训练量化Base，当前路径通常不适用 |

QLoRA 原论文围绕 4-bit 设计，所以 8-bit 分支没有 NF4/double-quant 组合；8-bit Base 与 LoRA 组合通常称为 8-bit LoRA。

NF4 是针对近似正态分布神经网络权重设计的非均匀 4-bit 表示，16 个表示值在 0 附近更密集。Double quantization 不是把权重重复量化，而是进一步量化第一次量化使用的缩放常数。

量化会引入近似误差，但显著降低 Base 权重显存。计算时权重按块反量化到 BF16/FP16 等计算精度，Base 冻结，梯度更新较高精度 LoRA 参数。

当前脚本不自动合并 LoRA。常见做法是重新以 BF16/FP16 加载原始 Base，再加载 adapter 并 `merge_and_unload()`，最终保存较高精度合并模型。

### 24.14 `prepare_model_for_kbit_training`

量化配置只决定“模型怎样量化加载”，并没有完成 LoRA 训练准备。因此量化加载后执行：

```python
model = prepare_model_for_kbit_training(
    model,
    training_args.gradient_checkpointing,
)
```

当前 PEFT 实现主要进行：

1. 将现有 Base 参数 `requires_grad=False`；
2. 把符合条件的非量化 FP16/BF16 参数提升为 FP32，提高稳定性；
3. 在开启 gradient checkpointing 时让 embedding 输出进入梯度图；
4. 启用 gradient checkpointing。

随后才执行：

```python
model = get_peft_model(model, peft_config)
```

插入新的可训练 LoRA 参数。因此顺序是：

```text
量化加载Base
→ 冻结Base并准备计算图
→ 插入requires_grad=True的LoRA A/B
```

### 24.15 `target_modules=all` 如何筛选模块

普通模型寻找：

```python
torch.nn.Linear
```

4-bit/8-bit 模型分别寻找：

```python
bnb.nn.Linear4bit
bnb.nn.Linear8bitLt
```

```python
isinstance(module, cls)
```

判断当前 `module` 对象是否是目标线性类或其子类实例。

模型中的模块不全是线性层，还包括：

- Embedding；
- RMSNorm/LayerNorm；
- Dropout；
- 激活函数；
- Attention组合模块；
- DecoderLayer；
- ModuleList；
- Linear。

`named_modules()` 递归遍历所有子模块。函数对匹配的线性层取名称最后一段，例如：

```text
model.layers.0.self_attn.q_proj → q_proj
```

再去重，并排除包含：

```text
lm_head
output_layer
```

的输出层。

`target_modules=all` 的真实含义是“在自动找到的线性层插入 LoRA”，不是全参数训练。若只想插入注意力层，可明确配置：

```bash
--target_modules q_proj,k_proj,v_proj,o_proj
```

`lm_head` 和 `output_layer` 通常承担 hidden state 到词表 logits 的最终映射，是不同架构中的常见不同命名；功能相似，但不保证永远是同一个 Python 对象。

### 24.16 类方法与模型对象

普通实例方法一般需要：

```python
obj = MyClass()
obj.method()
```

`@classmethod` 和 `@staticmethod` 可以通过类调用。`from_pretrained()` 是类级工厂方法，因此：

```python
model = AutoModelForCausalLM.from_pretrained(...)
```

会直接返回具体模型实例。

```python
**model_kwargs
```

将字典展开为关键字参数。例如：

```python
{"torch_dtype": torch.bfloat16, "device_map": "auto"}
```

等价于：

```python
from_pretrained(
    ...,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

### 24.17 混合精度与训练状态

混合精度表示：

- 大部分矩阵计算和激活使用 BF16/FP16；
- 数值敏感操作或输出可使用 FP32；
- 优化器状态/部分可训练参数可保留 FP32。

这样在速度、显存和稳定性之间折中。当前脚本还会把可训练参数转为 FP32，并为某些 `lm_head` 输出注册 FP32 hook。

### 24.18 Gradient checkpointing 与输入梯度

```python
if (
    training_args.gradient_checkpointing
    and getattr(model, "supports_gradient_checkpointing", False)
):
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
else:
    model.config.use_cache = True

model.enable_input_require_grads()
```

含义：

- 用户开启且模型支持时，启用激活重计算；
- `use_cache=False` 关闭训练中不需要且通常不兼容的 KV cache；
- 训练结束后重新开启 cache；
- `getattr(..., False)` 在属性不存在时安全返回 False。

`enable_input_require_grads()` 不会让整数 `input_ids` 可训练，也不会解冻 Base embedding。它通过 hook 让 embedding 输出的浮点 hidden states 设置 `requires_grad=True`，保证冻结 Base、PEFT 与 gradient checkpointing 组合时，梯度能继续传到 LoRA。

概念状态：

```text
input_ids：整数，无梯度
embedding.weight：冻结
embedding输出：requires_grad=True
LoRA参数：requires_grad=True
```

### 24.19 Dataset、DataLoader 与 DataCollator

tokenizer 编码在：

```python
train_dataset.map(preprocess_function, batched=True)
```

中完成。此后 `train_dataset` 已包含：

```python
input_ids
attention_mask
labels
```

DataLoader 负责：

- 从 Dataset 选样本；
- 训练时打乱；
- 按 batch size 分组；
- 调用 data collator；
- 迭代产生 batch。

`DataCollatorForSeq2Seq` 在本项目中主要用于：

- 动态 padding `input_ids`；
- padding `attention_mask`；
- labels 的 padding 位置填 `-100`；
- 转换为 PyTorch Tensor；
- 可选补齐到 4 的倍数。

名字带 Seq2Seq 不代表此处模型变成 encoder-decoder；项目只是复用它的动态 padding 能力。

```text
Dataset中的变长list
→ DataLoader选择一批
→ DataCollator动态补齐
→ Tensor batch
→ model
```

### 24.20 Trainer 的建立与前向传播

```python
trainer = SavePeftModelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)
```

Trainer 负责：

- 创建 DataLoader；
- 创建 optimizer 和 scheduler；
- 调用 `model(**batch)`；
- 读取 `outputs.loss`；
- backward；
- gradient accumulation；
- optimizer/scheduler step；
- 混合精度、DDP、日志、验证与保存。

真正启动训练：

```python
trainer.train(resume_from_checkpoint=checkpoint)
```

`resume_from_checkpoint` 可以是：

```text
outputs-sft-qwen-v1/checkpoint-500
```

用于恢复模型/adapter、optimizer、scheduler、global step 和随机状态。

```python
trainer.is_world_process_zero()
```

只在全局 rank 0 进程返回 True，用于避免多 GPU/多节点下重复打印和保存。

### 24.21 周期性 checkpoint 与最终保存

项目定义：

```python
class SavePeftModelTrainer(Trainer):
    def save_model(self, output_dir=None, ...):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, output_dir / TRAINING_ARGS_NAME)
        self.model.save_pretrained(output_dir)
```

训练过程中，当达到：

```bash
--save_strategy steps
--save_steps 500
```

Trainer 内部的 checkpoint 流程会调用这个 `save_model()`。

LoRA checkpoint 典型内容：

```text
checkpoint-500/
├── adapter_model.safetensors
├── adapter_config.json
├── training_args.bin
├── optimizer.pt
├── scheduler.pt
├── trainer_state.json
└── rng_state.pth
```

其中 adapter/config 由重写的 `save_model()` 保存；optimizer、scheduler、Trainer state 等由 Trainer 继承的 `_save_checkpoint` 流程负责。

训练结束后：

```python
trainer.save_state()
save_model(model, tokenizer, training_args)
```

全局 `save_model()` 会解包可能的分布式 wrapper，并保存最终模型/adapter和 tokenizer。

区别：

| 保存动作 | 位置/时间 | 主要作用 |
|---|---|---|
| `SavePeftModelTrainer.save_model()` | 周期性checkpoint | 保存当时模型/adapter |
| Trainer `_save_checkpoint` | 周期性checkpoint | 额外保存optimizer/scheduler/state |
| `trainer.save_state()` | 训练结束 | 保存Trainer状态 |
| 全局 `save_model()` | output根目录 | 保存最终模型/adapter和tokenizer |

目录通常是：

```text
outputs-sft-qwen-v1/
├── adapter_model.safetensors      # 最终推理结果
├── adapter_config.json
├── tokenizer.json
├── trainer_state.json
├── checkpoint-500/                # 可用于续训，也可加载中间adapter推理
└── checkpoint-1000/
```

根目录通常用于最终推理；`checkpoint-N` 含 optimizer/scheduler 等时适合断点续训。最终根目录不一定包含完整 optimizer 状态。

### 24.22 本文件的关键风险与检查项

1. 训练和验证目录当前相同，正式实验必须拆分；
2. `data/sft` 中所有 JSONL 都会混合加载，不会只训练 medical；
3. `cache_dir` 当前没有显式传给模型权重的 `from_pretrained()`；
4. 数据通过格式校验不代表医学事实或安全行为正确；
5. 新增特殊 token 时要检查模型 embedding 是否需要 resize；
6. `target_modules=all` 是所有匹配线性层，不是所有参数；
7. QLoRA 需要 `use_peft=True + load_in_4bit=True + qlora=True`；
8. 运行时要检查 target modules 和 trainable parameter 日志；
9. 训练前 decode 一条 `input_ids` 和处理后的 labels，确认模板、EOS 和 `-100`；
10. 保存后检查 adapter safetensors 的键名、shape 和 `modules_to_save`。

本文件的最终数据流总结：

```text
JSONL conversations
→ load_local_json_datasets
→ batched examples
→ 角色归一化和system/tools处理
→ chat template
→ dialog[source,target,...]
→ tokenizer.encode
→ input_ids/attention_mask/labels
→ DataCollator动态padding
→ Trainer调用具体Qwen/Llama forward
→ next-token loss
→ 梯度更新LoRA或全参数
→ adapter/full-model checkpoint
```

### 24.23 单卡 batch、micro-batch 与全局有效 batch

当前 `scripts/run_sft.sh` 使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 \
    training/supervised_finetuning.py \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

全局有效 batch 的计算是：

```text
global batch
= per_device_train_batch_size × world_size × gradient_accumulation_steps
= 2 × 2 × 8
= 32
```

这里应区分三个量：

| 概念 | 当前样本数 | 含义 |
|---|---:|---|
| 单卡 mini-batch | 2 | 每个 GPU/进程一次前向传播读取 2 条样本 |
| 多卡一次 micro-batch | 4 | 两张卡同时各处理 2 条，共处理 4 条 |
| 全局有效 batch | 32 | 连续累积 8 个 micro-batch 后更新一次参数 |

`gradient_accumulation_steps=8` 表示连续进行 8 次前向和反向传播，将梯度累积起来，第 8 次以后才执行一次 `optimizer.step()`，随后清空梯度。它用更多计算时间换取较低的单次显存峰值，但不会减少模型权重和优化器状态本身占用的显存。

Trainer 中的一个 `global_step` 通常对应一次优化器更新，而不是一次 micro-batch。因此当前配置下一次 `global_step` 理想上对应 32 条训练样本。

### 24.24 训练启动参数及所属位置

| 参数 | 所属位置 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Shell/CUDA 环境变量 | 控制当前程序能够看到哪些物理 GPU |
| `torchrun --nproc_per_node` | `torchrun` 启动参数 | 当前节点启动多少个训练进程 |
| `per_device_train_batch_size` | `Seq2SeqTrainingArguments` | 每个 GPU/进程一次读取的训练样本数 |
| `gradient_accumulation_steps` | `Seq2SeqTrainingArguments` | 累积多少个 micro-batch 后更新一次参数 |
| `model_max_length` | 项目自定义 `ScriptArguments` | 模板处理和分词后每条样本允许的最大 token 长度 |
| `num_train_epochs` | `Seq2SeqTrainingArguments` | 完整遍历训练集的次数 |
| `learning_rate` | `Seq2SeqTrainingArguments` | 优化器更新参数的基础步长 |
| `eval_steps` | `Seq2SeqTrainingArguments` | 当 `eval_strategy=steps` 时，每隔多少个更新步验证 |
| `save_steps` | `Seq2SeqTrainingArguments` | 当 `save_strategy=steps` 时，每隔多少个更新步保存 checkpoint |
| `output_dir` | `Seq2SeqTrainingArguments` | 最终模型和训练 checkpoint 的输出目录 |
| `bf16` | `Seq2SeqTrainingArguments` | 启用 BF16 混合精度训练 |
| `gradient_checkpointing` | `Seq2SeqTrainingArguments` | 反向传播时重算部分激活值，以计算时间换显存 |

`model_max_length` 限制的是 system、user、assistant 和特殊 token 组成的完整训练序列，不是最大新生成长度。`eval_steps` 和 `save_steps` 计算的是优化器更新步；如果总更新步数没有达到设定值，训练过程中就不会出现对应周期的验证或 checkpoint。

`gradient_checkpointing` 是激活重计算技术，与磁盘中的 `checkpoint-500/` 不是同一个概念：前者用于节省显存，后者用于保存模型和断点续训状态。

### 24.25 `run_sft.sh` 与直接输入终端命令

`run_sft.sh` 没有引入另一套训练逻辑。执行：

```bash
bash scripts/run_sft.sh
```

等价于把脚本内的完整命令复制到终端执行。真正训练的仍是 `training/supervised_finetuning.py`，双进程来自：

```bash
torchrun --nproc_per_node 2
```

而不是来自 `.sh` 文件本身。Shell 脚本的主要价值是：

1. 保存完整实验参数；
2. 减少手动输入和漏参；
3. 方便重复运行和 Git 版本管理；
4. 让实验配置容易复现。

`torchrun` 后面直接写训练脚本即可：

```bash
torchrun --nproc_per_node 2 training/supervised_finetuning.py ...
```

一般不写成 `torchrun ... python training/supervised_finetuning.py`，因为 `torchrun` 自己负责启动 Python 训练进程。

脚本中的 `./data/sft`、`./cache` 和 `outputs-sft-qwen-v1` 都是相对于执行命令时的工作目录，所以应从项目根目录执行：

```bash
cd /home/gycing/llm/MedicalGPT
bash scripts/run_sft.sh
```

### 24.26 `CUDA_VISIBLE_DEVICES=0,1 python ...` 到底使用哪张卡

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python training/supervised_finetuning.py ...
```

准确含义只是：

1. 启动一个 Python 进程；
2. 让该进程能够看到物理 GPU 0 和 GPU 1；
3. 两张物理卡会在进程内部重新编号为逻辑 `cuda:0` 和 `cuda:1`。

它不表示“自动任选一张”，也不表示自动建立两个进程的数据并行。最终设备使用方式取决于程序：

- 程序使用默认 CUDA 设备时，一般使用逻辑 `cuda:0`，即这里的物理 GPU 0；
- 程序显式指定 `cuda:1` 时使用第二张可见卡；
- 程序使用 `device_map="auto"` 时，可能把模型的不同层分布到两张卡上；
- 只有使用 `torchrun --nproc_per_node 2` 等分布式启动方式，才会建立两个训练进程进行标准 DDP 数据并行。

本项目的 `ModelArguments.device_map` 默认是 `"auto"`。当只有一个进程而且看见多张 GPU 时，代码允许 Hugging Face 自动安排模型层：模型较小时可能主要放到一张卡，模型较大时可能拆到两张卡。这属于单进程模型分片，不是双进程 DDP。

当使用 `torchrun` 时，代码检测到 `WORLD_SIZE != 1`，会将 `device_map` 设为 `None`，再由分布式训练机制让每个进程负责自己的 GPU：

```text
进程 0 → 一份模型副本 → GPU 0 → 不同数据
进程 1 → 一份模型副本 → GPU 1 → 不同数据
                         ↓
                    同步梯度更新
```

所以要根据目标选择：

```text
限制程序只用一张卡：CUDA_VISIBLE_DEVICES=0 python ...
单进程自动模型分片：CUDA_VISIBLE_DEVICES=0,1 python ...（配合 device_map=auto）
双卡数据并行训练：  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 ...
```

### 24.27 `glaive_toolcall_zh_demo.jsonl` 的作用

`medical_sft_1K_format.jsonl` 和 `sharegpt_zh_1K_format.jsonl` 主要训练“用户提问 → assistant 回答”；单轮和多轮数据可以训练聊天能力，但不能系统教会模型遵循工具调用协议。

`glaive_toolcall_zh_demo.jsonl` 主要包含：

```text
human
→ function_call
→ observation
→ gpt最终回答
```

各字段的作用：

| 字段/角色 | 含义 | 训练中的位置 |
|---|---|---|
| `tools` | 工具名称、说明、参数schema和必填字段 | 被格式化为system工具说明 |
| `human` | 用户提出的任务 | 输入，默认不参与回答loss |
| `function_call` | 模型选择的工具和JSON参数 | assistant目标，参与loss |
| `observation` | 外部程序执行工具后的返回值 | 环境输入，默认不参与回答loss |
| `gpt` | 模型根据工具结果整理的最终回答 | assistant目标，参与loss |

`supervised_finetuning.py` 会把 `function_call` 映射为 assistant，把 `observation` 格式化后映射为 user，并根据 `--tool_format` 生成对应模型家族的调用格式。

工具调用数据只训练“何时调用、选择哪个工具、生成什么参数、怎样使用返回结果”，不会创建真实工具，也不会让模型自动联网。推理系统仍需负责解析调用、执行Python函数或API、把结果作为observation送回模型。

是否需要这类数据取决于目标：纯医疗问答可以不使用；要做指南检索、药物相互作用查询或风险评分等医疗Agent，就需要高质量的医疗工具调用数据。通用发票、新闻样本不应因为数量多就直接混进医疗正式训练。

当前示例若以整个 `data/sft` 为训练目录，会同时加载medical、sharegpt和glaive三个JSONL。小模型容量有限，正式实验应明确数据组成，不能把“目录名是sft”当成任务一致。

### 24.28 MedAegis SFT smoke 数据和配置

Day 3 的独立小实验位于：

```text
notes/day3/test/
├── run_sft.sh
├── sft_smoke_config.json
└── data/
    ├── train/sft_smoke_train.jsonl   # 24条
    └── dev/sft_smoke_dev.jsonl       # 10条（含2条多轮）
```

train和dev的ID、首轮问题互不重叠。样本覆盖医疗知识、信息不足、危险信号、用药安全、多轮信息补充和非医疗问题。它们用于验证数据→token→loss→LoRA→评估→保存链路，不是经过临床审核的医学金标准。

所有样本使用统一的简短system prompt，项目实验名使用 **MedAegis**。0.8B smoke配置为：

```text
单卡/单进程
per-device train batch = 1
gradient accumulation = 4
effective batch = 4
model_max_length = 256
train/dev = 24/10
epoch = 1
FlashAttention = false
```

用当前Qwen tokenizer检查时，训练样本最长约160 tokens，验证样本最长约171 tokens，均未超过256。该检查只能说明当前样本不会因长度上限截断，不能替代训练时decode一条input和labels。

### 24.29 `run_sft.sh` 与 `sft_smoke_config.json` 的关系

`run_sft.sh` 是实际执行入口，参数会被 `HfArgumentParser` 传进数据类；`sft_smoke_config.json` 当前只是便于审计和复现的实验清单，不会被训练脚本自动读取。

因此：

- `num_train_epochs=1` 同时记录在两处，真正生效的是shell传入的 `--num_train_epochs 1`；
- `effective_train_batch_size=4` 是记录用的派生结果，不是Python命令行参数；
- 如果修改shell参数，必须同步更新JSON，否则记录会与真实运行不一致；
- 最终应以实际命令、日志和输出目录中的训练参数文件为准。

有效batch的来源是：

```text
effective batch
= per_device_train_batch_size × world_size × gradient_accumulation_steps
= 1 × 1 × 4
= 4
```

不能把 `--effective_train_batch_size 4` 直接加到命令中，因为当前数据类没有这个字段，解析器会把它视为未知参数。

### 24.30 smoke训练的日志、验证和保存参数

当前24条训练样本、有效batch为4、训练1个epoch，理想情况下约有：

```text
24 ÷ 4 = 6个optimizer steps
```

这里的Trainer `global_step` 通常是一次 `optimizer.step()`，不是一次单样本前向传播。

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `warmup_steps` | 1 | 第1个优化器更新步进行学习率预热 |
| `logging_strategy` | `steps` | 按优化器更新步记录日志 |
| `logging_steps` | 1 | 每一步记录loss、学习率等指标 |
| `eval_strategy` | `steps` | 按更新步执行验证 |
| `eval_steps` | 2 | 第2、4、6步附近验证 |
| `save_strategy` | `steps` | 按更新步保存checkpoint |
| `save_steps` | 3 | 第3、6步附近保存checkpoint |
| `save_total_limit` | 2 | 最多保留两个周期性checkpoint |
| `TENSORBOARD_LOGGING_DIR` | `.../tensorboard` | 在Transformers 5.17中固定TensorBoard event文件目录的环境变量 |
| `report_to` | `tensorboard` | 启用Trainer的TensorBoard集成 |

`eval_steps` 只有配合 `eval_strategy=steps` 才控制周期验证；`save_steps` 同理需要 `save_strategy=steps`。当前环境的Transformers 5.17.0没有 `Seq2SeqTrainingArguments.logging_dir` 字段，传入 `--logging_dir` 会被 `HfArgumentParser` 判为未使用参数并报错。脚本改用 `TENSORBOARD_LOGGING_DIR` 环境变量；它不是模型权重目录，真正的模型和checkpoint写入 `output_dir`。若不设置该环境变量，TensorBoard回调默认写到 `output_dir/runs/...`。

这些策略大多可以依赖Transformers默认值，但为了复现，实验脚本应显式写清。对于smoke test，每步日志、每两步验证是为了快速暴露loss、数据或评估链路问题，不是正式训练的性能最优配置。

### 24.31 LoRA精度相关参数

`lora_dropout=0.05` 表示训练时在LoRA支路以5%的概率丢弃部分输入特征，用于正则化；它不是删除5%的样本，也不是修改Base模型所有层的dropout。验证和推理时dropout关闭。

简化表达为：

```text
y = W x + (alpha / rank) × B A Dropout(x)
```

其中 `W` 是冻结的Base权重，`A/B` 是可训练LoRA矩阵。

`--torch_dtype bfloat16` 主要控制模型加载的数据类型；`--bf16` 告诉Trainer启用BF16混合精度训练。二者相关但不完全相同，不能据此认为所有权重、激活、优化器状态和数值敏感操作都强制为BF16。

### 24.32 `max_train_samples` 与预处理进程

`max_train_samples` 是原始训练记录数量的上限，不是token数、batch数或训练步数。代码先shuffle，再取：

```python
max_train_samples = min(len(train_dataset), configured_limit)
train_dataset = train_dataset.select(range(max_train_samples))
```

当前文件有24条并设置 `max_train_samples=24`，所以预处理前最多选择24条。tokenizer处理后，如果某些样本无效或labels全为 `-100` 而被过滤，真正参与训练的数量仍可能少于24。`max_eval_samples` 对验证集同理。当前实现中不传上限或设置非正数时不会执行正数截取。

`preprocessing_num_workers=2` 表示datasets预处理使用两个操作系统进程，不是服务器拥有“两块CPU”。操作系统可能把两个进程调度到不同CPU核心。它只用于模板处理、tokenizer编码和过滤等CPU预处理，不是两个GPU训练进程。

对于24+10条smoke数据，多进程启动和合并的开销可能比收益更大，设置1或使用默认值即可；正式处理几千、几万条数据时再根据CPU核心、内存和实测吞吐提高 `num_proc`。

### 24.33 项目品牌与开源框架来源

个人实验品牌使用 **MedAegis**，可描述为“面向医疗大模型的风险感知与证据约束对齐系统”。`MedicalGPT` 是当前使用和学习的上游训练仓库。品牌重命名不能等同于从零实现训练框架；面试和发布时应明确个人完成的数据治理、安全任务设计、SFT/DPO实验、评测和bad-case分析，同时保留所复用开源代码的许可证与来源说明。

最终可以建立独立的MedAegis交付仓库，只迁移真实使用和改造的模块；不应为了隐藏依赖而删除Apache 2.0许可证或误报框架来源。

---

## 原始学习记录：数据应用对象

* 文本：fever.txt主要用于增量预训练

*第一章论
传染病是指由病原微生物，如朊粒、病毒、衣原体、立克次体、支原体（mycoplasma)细菌真菌、螺旋体和寄生虫，如原虫、蠕虫、医学昆虫感染人体后产生的有传染性、在一定条件下可造成流行的疾病。感染性疾病是指由病原体感染所致的疾病，包括传染病和非传染性感染性疾病。
传染病学是一门研究各种传染病在人体内外发生、发展、传播、诊断、治疗和预防规律的学科。重点研究各种传染病的发病机制、临床表现、诊断和治疗方法，同时兼顾流行病学和预防措施的研究，做到防治结合。
传染病学与其他学科有密切联系，其基础学科和相关学科包括病原生物学、分子生物学、免疫学、人体寄生虫学、流行病学、病理学、药理学和诊断学等。掌握这些学科的基本知识、基本理论和基本技能对学好传染病学起着非常重要的作用。*

* conversations: 对话格式，监督微调

*{"conversations":[{"from":"human","value":"精子生成减少的病因是什么？"},{"from":"gpt","value":"支持细胞错位"}]}*

*`{"conversations":[{"from":"human","value":"两只脚明显大小不一样，腿也不一样粗，该怎么办，两只脚明显大小不一样，腿也不一样粗，该怎么办，需要做什么检查"},{"from":"gpt","value":"，与走路姿势没有关系的，人的器官，没有完全对称的，只是有的不是很明显的，这很正常的，只要健康就好。只有手术能纠正的。"}]}`*

* chosen/rejected: 强化学习，RLHF,reward？dpo

```jsonl
{"conversations": [{"from": "human", "value": "气候变化正日益成为全球关注的焦点，随着温度持续上升和极端天气事件创纪录地发生。这是由于人类行为造成的温室气体排放，如二氧化碳，这些气体会捕获热量并像一层毯子一样包裹在地球大气层周围。气候变化的影响已经开始显现，从洪水到干旱，如果气候继续升温，预计情况将会进一步恶化。因此，人们有必要通过可持续和可再生能源以及其他积极措施来减少温室气体的排放。"}], "chosen": "气候变化是全球日益关注的问题，随着温度持续上升。这是由于人类行为，如排放二氧化碳等温室气体，这些气体会捕获热量并像一层毯子一样包裹在地球大气层周围。气候变化的影响已经在洪水、干旱和其他极端天气事件中得到体现。应对气候变化需要通过可持续和可再生能源减少温室气体的排放，以及其他积极的措施。不采取行动可能导致气候条件恶化及其破坏性影响。", "rejected": "气候变化是由人类活动引起的严重问题，如化石燃料的燃烧。这导致地球温度上升，引发了各种负面影响，如更极端的天气事件、海平面上升以及植物和动物生活的变化。\n\n有许多方法可以应对气候变化。其中最重要的之一是减少温室气体排放。这可以通过使用可再生能源，如太阳能和风能，以及提高能源效率来实现。种植树木也很重要，因为它们可以吸收大气中的二氧化碳。\n\n气候变化是一个复杂的问题，但我们必须加以解决。通过立即采取行动，我们可以帮助保护我们的地球，造福后代。\n\n* 气候变化是由温室气体排放到大气中引起的。这些气体会捕获热量，导致地球温度上升。\n* 主要的温室气体包括二氧化碳、甲烷和氧化亚氮。这些气体来自各种来源，包括燃烧化石燃料、森林砍伐和农业。\n* 气候变化引发了各种负面影响，如更极端的天气事件、海平面上升以及植物和动物生活的变化。\n* 有许多方法可以应对气候变化。其中最重要的之一是减少温室气体排放。这可以通过使用可再生能源，如太阳能和风能，以及提高能源效率来实现。种植树木也很重要，因为它们可以吸收大气中的二氧化碳。\n* 气候变化是一个复杂的问题，但我们必须加以解决。通过立即采取行动，我们可以帮助保护我们的地球，造福后代。"}
{"conversations": [{"from": "system", "value": "你是一个非常聪明的AI助手，非常擅长按照指示行事。尽你所能地帮助。"}, {"from": "human", "value": "请为以下电影情节提供一个电影标题：1964年的一个简短场景中，年迈肥胖的意大利裔美国人杰克·拉莫塔（罗伯特·德尼罗饰）练习喜剧表演。然后整部电影发生在回忆中。1941年，拉莫塔在一场重要的拳击比赛中对阵吉米·里夫斯，他在比赛中首次失利。杰克的兄弟乔伊·拉莫塔（乔·佩西饰）与他的黑手党联系人之一萨尔维·巴茨（弗兰克·文森特饰）讨论了一个有可能争夺中量级冠军头衔的机会。此后不久，杰克在布朗克斯社区的一个露天游泳池里看到一个名叫维基（凯西·莫里亚蒂饰）的15岁女孩。尽管他已经结婚，但最终他与她开始了一段关系。1943年，杰克击败了糖·雷·罗宾逊，并在三周后进行了一场复赛。尽管杰克在比赛中占据上风，但裁判出人意料地判罗宾逊获胜，乔伊认为罗宾逊只是因为下周要入伍到美国陆军才赢得了比赛。到了1947年，杰克与维基结婚。杰克经常担心维基对其他男人有感情，尤其是当她对杰克下一场比赛的对手托尼·贾尼罗做出不经意的评论时。当他在当地黑帮老大汤米·科莫（尼古拉斯·科拉桑托饰）和维基面前残忍地打败贾尼罗时，他的嫉妒心显而易见。当乔伊在科帕卡巴纳与记者们讨论胜利时，他被看到维基走向萨尔维和他的团队的桌子而分心。乔伊与维基交谈，维基说她已经对他的兄弟失望了。乔伊责怪萨尔维，激烈地袭击他，打斗甚至蔓延到俱乐部外。科莫后来命令他们道歉，并让乔伊告诉杰克，如果他想有机会争夺由科莫控制的冠军头衔，他必须先认输。在与比利·福克斯的比赛中，杰克在短暂地殴打对手后，甚至都不费力地进行抵抗。他随后因涉嫌故意输掉比赛而被暂时停赛，尽管他意识到自己判断错误时已经太迟。最终他被恢复资格，并在1949年赢得了..."}], "chosen": "\"愤怒的传奇：杰克·拉莫塔的故事\"", "rejected": "根据您提供的电影情节，这里有一些潜在的电影标题想法：\n\n1. \"布朗克斯的公牛：爱情、失落和皮革的故事\"\n2. \"愤怒的公牛：杰克·拉莫塔的未被揭示的故事\"\n3. \"爱与拳拳：杰克·拉莫塔的故事\"\n4. \"中量级黑帮分子：拳击剧\"\n5. \"科帕卡巴纳团队：拳击、背叛和兄弟之爱的故事\"\n6. \"拳击手的心：嫉妒、贪婪和救赎的故事\"\n7. \"游泳池的罗密欧：拳击手的爱情与失落的故事\"\n8. \"黑手党的媒人：拳击喜剧剧情片\"\n9. \"布朗克斯拳击手：激情、力量和拳击的故事\"\n10. \"愤怒的心：杰克·拉莫塔传记片\"\n\n希望这些建议对您有所帮助！如果需要进一步的帮助，请告诉我。"}
{"conversations": [{"from": "human", "value": "探讨有机农业实践的历史和演变。"}], "chosen": "有机农业实践的演变：走向可持续发展之路\n\n介绍\n\n近年来，有机农业实践引起了广泛关注，消费者越来越意识到传统农业方法对环境和健康的影响。然而，有机农业的根源可以追溯到古代文明，那时农业与自然和谐共处。在这篇博客文章中，我们将探讨有机农业实践的历史和演变，以及它们如何成为可持续农业运动的重要组成部分。\n\n早期起源：古代有机农业实践\n\n最早记录的有机农业实践证据可以在古代文明中找到，如埃及人、希腊人和罗马人。这些社会认识到保持土壤健康和生物多样性的重要性，并采用了轮作、堆肥和多种栽培等技术来实现这一目标。\n\n例如，古埃及人使用了一种盆地灌溉系统，使他们能够在尼罗河三角洲种植作物。他们还采用了轮作技术，如在种植小麦和大麦之间交替，以保持土壤肥力。同样，古希腊人和罗马人实行多种栽培，他们在同一块土地上种植多种作物，并利用引入有益昆虫来控制害虫等自然防治方法。\n\n中世纪和单一栽培的兴起\n\n在中世纪，农业实践开始向单一栽培转变，即在大片土地上种植单一作物。这在很大程度上是由于新的农业技术的引入，如重犁，它使土地的耕作更加高效。然而，这种向单一栽培农业的转变导致了土壤退化和侵蚀，以及生物多样性的丧失。\n\n现代时代和有机农业运动\n\n20世纪20年代和30年代，有机农业的现代时代开始了，阿尔伯特·霍华德和J.I.罗代尔等先驱的工作起到了关键作用。他们倡导回归自然和可持续的农业实践，并发展了堆肥、覆盖作物和轮作等方法。\n\n在20世纪60年代和70年代，有机农业运动获得了势头，英国的土壤协会和美国的国家有机计划等组织的建立推动了这一进程。这些组织制定了有机农业的标准和认证计划，以确保农场遵守可持续实践。\n\n如今，有机农业实践被认为是可持续农业的关键组成部分，对有机产品的需求不断增长。有机农民采用的自然害虫控制方法、保护耕作和多种栽培等技术只是促进土壤健康、生物多样性和生态平衡的几个例子。\n\n有机农业实践的未来\n\n随着世界人口的持续增长，对可持续和环保意识的农业实践的需求只会增加。有机农业实践的未来看起来很有前途，技术和研究的进步导致了更高效和创新的方法。例如，精准农业的应用越来越广泛，它涉及使用传感器、无人机和卫星成像来监测和管理作物生长。\n\n此外，城市农业和社区支持的农业计划的兴起为人们提供了获取新鲜、当地种植的农产品的新机会。这些计划还有助于促进社区参与和教育，推动可持续农业实践。\n\n结论\n\n有机农业实践的历史和演变反映了朝着更可持续和环保意识方法不断前进的旅程。从最早采用自然农业技术的古代文明到推动可持续农业的现代组织，有机农业实践在保护地球健康方面发挥了至关重要的作用。展望未来，有机农业实践将继续在确保后代获得健康、可持续食品供应方面发挥关键作用。", "rejected": "从古老的根源到现代运动：有机农业实践的历史和演变\n\n有机农业自其作为一种农业实践的谦逊开端以来已经走过了漫长的道路。随着对环境、人类健康和可持续发展的日益关注，有机农业现在正在经历一场复兴，将农业带回到其根源。\n\n虽然有些人可能认为有机农业是对工业农业的现代运动的回应，但事实是，其历史可以追溯到数千年前，当时古代文明实践了与自然和谐的可持续农业技术。\n\n在这篇博客文章中，我们将通过有机农业实践的历史和演变之旅，探索其古老起源、工业化的影响以及我们今天看到的有机运动的最终崛起。\n\n古老起源和传统实践：\n\n有机农业并不是一个新概念，其根源可以追溯到古代文明实践的最早形式的农业。许多传统农业系统，如印度吠陀农业、古埃及农业或美洲土著农业实践，都专注于培育土壤肥力和维持生态系统内的平衡。\n\n这些技术通常包括轮作、堆肥和使用天然防治害虫剂，以与自然和谐的方式。我们现在认识为有机农业原则的东西实际上对这些早期社会来说是一种自然的生活方式。\n\n工业化的影响：\n\n18世纪末工业革命的开始导致了农业实践的转变。随着技术进步增加生产力，商业化和农业的工业化得以推动。合成肥料、化学农药和大规模单一作物种植系统开始取代传统农业方法，破坏了几个世纪来一直保持的土壤生态系统和微妙的平衡。\n\n这种实践转变导致了增加的作物产量，但代价是环境退化、生物多样性减少以及对人类健康的合成化学品影响的担忧。\n\n有机运动的诞生：\n\n20世纪初，一个反运动开始形成，农业科学家和环保主义者开始担心化学农业的后果。阿尔伯特·霍华德爵士、伊芙·巴尔弗夫夫人和J.I.罗代尔等知名人物为有机农业原则的重新发现和重新定义做出了贡献。他们的研究和倡导奠定了我们现在认识的现代有机运动的基础。\n\n20世纪60年代，公众对环境的意识显著增强，部分原因是瑞秋·卡森的著作《寂静的春天》的出版，该书概述了合成农药的有害影响。这个十年标志着消费者对有机产品的需求增加以及英国的土壤协会和美国的有机材料审查研究所等有机认证计划的建立。\n\n今天的有机农业实践遵循着以土壤肥力、生物多样性和可持续性为核心的原则，将传统知识与现代农业进步相结合。有机农业不再仅依赖于过去几代人的体力劳动，而是整合了创新工具和技术，以优化资源利用并减少对环境的负面影响。\n\n认证机构和政府机构监管有机农业，确保严格遵守合成物质使用、土壤和水资源保护、生物多样性保护和动物福利的指导方针。\n\n有机农业自其古老起源以来已经走过了漫长的道路。通过将传统实践与现代知识相结合，它为传统农业提供了一种可持续的替代方案，确保更健康的生态系统、更具韧性的社区和更有营养的作物，使我们的星球和我们的餐桌都受益。"}
```
