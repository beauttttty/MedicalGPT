##### 模型推理

- tokenizer 在哪里加载？
- model 在哪里加载？
- LoRA 在 Base model 之前还是之后加载？
- 用户输入如何变成完整 prompt？
- `attention_mask` 有什么作用？
- 为什么生成结果要去掉输入 prompt？
- `temperature=0` 与 `0.7` 有什么不同？
- `max_new_tokens` 为什么不是最大输入长度？

## 推理：流式生成与批量生成

### 1. 两种模式的共同点

流式生成和批量生成底层都调用 `model.generate()`，模型处理的始终是 token ID，而不是直接处理字符串。两者的核心生成过程相同：

```text
文本
  ↓ tokenizer
input_ids
  ↓ 模型前向传播
logits
  ↓ 采样或 argmax
下一个 token ID
  ↓ 追加到已有序列后继续生成
最终 token ID 序列
  ↓ tokenizer 解码
文本回答
```

两者的主要区别不在于是否有系统提示词或是否使用会话模板，而在于：

- 当前一次处理多少个样本；
- 生成出来的 token ID 是立即解码，还是等待整批生成结束后统一解码；
- 使用场景更重视首字延迟还是整体吞吐量。

### 2. 当前项目中的流式生成

`stream_generate_answer()` 接收一个已经构造好的 `prompt`：

```python
inputs = tokenizer(prompt, return_tensors="pt")
```

这里的 `prompt` 是一个字符串，对应一个样本，所以：

```text
input_ids.shape = [1, sequence_length]
```

其中 batch size 为 1。但这个 prompt 并不一定只包含用户当前的问题。在调用该函数之前，交互模式已经完成以下工作：

1. 添加可选的 system prompt；
2. 添加以前的用户问题和助手回答；
3. 添加当前用户问题；
4. 调用 `tokenizer.apply_chat_template()` 生成完整 prompt。

因此，流式函数没有单独接收 `system_prompt`，是因为系统提示和历史对话已经被包含在传入的 prompt 中。

流式生成使用：

```python
streamer = TextIteratorStreamer(...)
```

模型仍然先生成 token ID，`streamer` 接收这些 ID，并在生成过程中增量解码：

```text
模型生成一部分 token ID
        ↓
streamer 缓存并解码
        ↓
for new_text in streamer
        ↓
立即显示文本片段
```

`new_text` 每次都是字符串，但它不是一个样本，也不一定对应一个 token。它只是当前能够稳定解码出的文本片段。多次循环得到的 `new_text` 共同组成同一个样本的一条完整回答：

```python
generated_text += new_text
```

当前代码将 `model.generate()` 放在线程中运行，是为了让模型生成和主线程读取 `streamer` 同时发生，从而实现边生成边显示。

### 3. 当前项目中的批量生成

`batch_generate_answer()` 接收多个问题：

```python
sentences = [
    "介绍下北京",
    "乙肝和丙肝的区别？",
]
```

函数会为每个问题分别构造消息：

```python
messages = [{"role": "user", "content": s}]
```

如果设置了系统提示，还会在消息列表开头加入：

```python
{"role": "system", "content": system_prompt}
```

每个消息列表分别经过：

```python
tokenizer.apply_chat_template(...)
```

得到多个 prompt。随后统一分词：

```python
inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
```

假设 batch 中有 4 个样本，则输入形状类似：

```text
input_ids.shape = [4, padded_sequence_length]
```

不同问题长度不同，因此要用 padding 补到相同长度。当前 tokenizer 使用左侧 padding，适合 decoder-only 因果语言模型的批量生成。

批量模式没有传入 streamer，而是等待 `model.generate()` 完成后获得完整 token ID 张量：

```text
outputs.shape ≈ [batch_size, padded_input_length + max_generated_length]
```

对于 decoder-only 模型，`outputs` 同时包含输入 prompt 和新生成的 token。因此代码先切掉输入部分：

```python
gen_sequence = gen_sequence[prompt_len:]
```

然后逐条解码：

```python
gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=True)
```

一批样本可能在不同时间生成 EOS。由于张量必须是规则矩形，较早结束的样本通常会在其他样本继续生成时被 padding 到统一长度。

### 4. 为什么一个常用 batch=1，另一个常用 batch>1

这不是模型的硬性限制，而是使用场景造成的工程选择：

| 模式     | 主要目标                          | 常见 batch size | 典型场景                       |
| -------- | --------------------------------- | --------------: | ------------------------------ |
| 流式生成 | 降低首字延迟，让用户尽快看到回答  |               1 | 命令行聊天、网页聊天、流式 API |
| 批量生成 | 提高 GPU 吞吐量，一次处理更多问题 |          大于 1 | 离线评测、批量预测、数据生成   |

从技术上说，批量生成也可以令 batch size 等于 1；流式生成也可以扩展为多样本，但必须区分每段输出属于哪个样本，代码会复杂很多。当前 `TextIteratorStreamer` 的用法主要面向单样本生成。

### 5. logits 如何变成 token ID

因果语言模型前向传播的输出通常为：

```text
logits.shape = [batch_size, sequence_length, vocab_size]
```

其含义是：对 batch 中每个样本的每个序列位置，模型都给整个词表中的每个 token 一个分数。

生成下一个 token 时，只需要当前最后位置的预测：

```python
next_token_logits = logits[:, -1, :]
```

形状变为：

```text
[batch_size, vocab_size]
```

然后按照生成策略选择每个样本的下一个 token：

- `do_sample=False` 时，通常选择分数最大的 token，即 argmax；
- `do_sample=True` 时，将 logits 转换成概率分布，再根据概率采样；
- `temperature`、`repetition_penalty` 等参数会在选择前调整 logits。

选出的结果为：

```text
next_token_ids.shape = [batch_size]
```

把它扩展为 `[batch_size, 1]` 并追加到现有 `input_ids` 后，再进行下一轮生成：

```text
[B, S] + [B, 1] → [B, S + 1]
```

因此，生成不是把整个 `[B, S, V]` 一次性变成完整的 `[B, S]` 回答，而是一次选出一个新 token，并进行自回归循环。实际的 `model.generate()` 通常会使用 KV Cache；首次前向计算处理整个 prompt，之后只需处理最新 token，对应的 logits 可能为 `[B, 1, V]`，但逻辑上仍然依赖完整历史。

流式模式与批量模式从 logits 选择 token ID 的过程完全相同，区别只发生在 token ID 选出以后：

```text
流式模式：每产生一部分 token ID，就交给 streamer 增量解码
批量模式：在 generate() 内累计 token ID，全部结束后统一返回并解码
```

### 6. EOS、`stop_str` 与 `skip_special_tokens`

模型通常在回答末尾生成 EOS（End Of Sequence）token。真正控制 `model.generate()` 停止的通常是：

- `eos_token_id`；
- `max_new_tokens`；
- Transformers 的 `stopping_criteria`；
- generation config 中的其他停止设置。

项目中的 `stop_str` 是解码后的字符串截断条件。批量模式会对每个样本分别执行：

```python
pos = gen_text.find(stop_str)
if pos != -1:
    gen_text = gen_text[:pos]
```

所以一批样本共用同一个 `stop_str`，但每条回答单独查找和截断。它主要是生成后的文本清理，不一定能让模型停止计算。

`skip_special_tokens=True` 不是忽略所有特殊字符，例如 `<`、`>` 或 `/`，而是删除 tokenizer 中明确注册的特殊 token，例如 EOS、BOS 和 PAD。如果 `</s>` 已注册为 EOS 特殊 token，那么解码时它通常已经被删除，此后再查找默认的 `stop_str="</s>"` 很可能找不到，因此该默认检查往往只是兼容性或遗留保护。

如果用户设置的 `stop_str` 是普通文本，例如：

```text
User:
```

它不是 tokenizer 注册的特殊 token，就不会被 `skip_special_tokens=True` 删除，此时字符串截断仍然有意义。

当前流式代码只在单个 `new_text` 片段中查找停止字符串。如果停止字符串跨越两个片段，例如第一段以 `</` 结尾、第二段为 `s>`，就无法找到完整的 `</s>`。此外，即使读取端遇到字符串后 `break`，后台的 `model.generate()` 线程也可能继续运行，直到遇到 token 级停止条件或达到 `max_new_tokens`。若要可靠地停止实际生成，应优先使用 EOS token 或 token 级 `stopping_criteria`。

### 7. 最终对照

| 对比项               | 流式生成                  | 批量生成                     |
| -------------------- | ------------------------- | ---------------------------- |
| 当前实现的输入样本数 | 1                         | 通常大于 1                   |
| system prompt        | 调用前已加入完整 prompt   | 函数内部加入每个样本         |
| chat template        | 调用前应用                | 函数内部应用                 |
| 模型输入             | token ID                  | token ID                     |
| 模型原始输出         | logits                    | logits                       |
| token 选择           | `generate()` 内逐步选择 | `generate()` 内逐步选择    |
| 解码时机             | 生成过程中增量解码        | 整批生成结束后逐条解码       |
| 返回方式             | 多个文本片段组成一条回答  | 返回完整 token ID 张量后解码 |
| 主要优势             | 首字延迟低、交互体验好    | GPU 吞吐量高                 |
| 常见用途             | 在线聊天、流式 API        | 离线评测、批量预测           |

### 8.执行流程

读取命令行参数
    ↓
加载 tokenizer
    ↓
加载基础模型
    ↓
可选：以 8 bit / 4 bit 加载
    ↓
可选：加载 LoRA 适配器
    ↓
model.eval() 切到推理状态
    ↓
┌──────────────────────┬──────────────────────┐
│ --interactive        │ 未使用 --interactive │
│ 终端多轮流式对话      │ 文件/示例批量推理      │
│ TextIteratorStreamer │ model.generate 批处理 │
└──────────────────────┴────

## 对话模板：template
