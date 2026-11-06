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

## 对话模板：`training/template.py`

### 1. 这个文件解决什么问题

`template.py` 的作用是把结构化对话转换成特定模型认识的文本格式：

```text
system / user / assistant 消息
              ↓
       Conversation 模板
              ↓
带角色标记、分隔符和特殊 token 的 prompt
              ↓
           tokenizer
              ↓
           input_ids
```

不同模型使用的角色标记可能不同。Qwen 使用 ChatML：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

因此，chat template 虽然最终表现为字符串，但不是可以随意拼接的普通字符串。模型在训练中学习了角色标记、换行、消息边界和结束标记的固定排列方式，推理时应保持同样的格式。

### 2. 三种角色

| 角色          | 作用                         | 在训练中的意义                       |
| ------------- | ---------------------------- | ------------------------------------ |
| `system`    | 规定模型身份、规则和回答风格 | 作为上下文条件，通常位于对话最前面   |
| `user`      | 用户问题或指令               | 模型根据它生成回答                   |
| `assistant` | 助手回答                     | SFT 中通常是需要预测、计算损失的目标 |

例如：

```python
messages = [
    {"role": "system", "content": "你是一名严谨的医疗助手"},
    {"role": "user", "content": "发烧应该怎么办？"},
    {"role": "assistant", "content": "建议先测量体温……"},
]
```

角色不能随意互换。把 `user` 和 `assistant` 对调，相当于训练模型生成问题而不是回答问题。

### 3. `Conversation` 类

`Conversation` 使用 `@dataclass` 定义。`@dataclass` 会根据类中的字段自动生成初始化方法，所以创建模板时可以直接传入字段：

```python
@dataclass
class Conversation:
    name: str
    system_prompt: str
    messages: Optional[List[Sequence[str]]]
    roles: Optional[Sequence[str]]
    prompt: str
    sep: str
    stop_str: Optional[str] = "</s>"
```

各字段含义：

| 字段              | 含义                                            |
| ----------------- | ----------------------------------------------- |
| `name`          | 模板名称，例如`qwen`                          |
| `system_prompt` | 默认系统提示词，可能已经包含模型专用的角色标记  |
| `messages`      | 对话历史，格式为`[[用户问题, 助手回答], ...]` |
| `roles`         | 模板声明的角色名                                |
| `prompt`        | 每一轮用户问题的格式，`{query}` 是问题占位符  |
| `sep`           | 轮次之间使用的文本分隔符                        |
| `stop_str`      | 文本层面的停止字符串，默认是`</s>`            |

当前 `_format_example()` 没有读取 `roles` 字段。实际的 `system`、`user` 和 `assistant` 标记已经直接写在 `system_prompt` 和 `prompt` 中，因此这里的 `roles` 主要起描述作用。

`append_message()` 将一轮消息保存到对象中：

```python
def append_message(self, query, answer):
    self.messages.append([query, answer])
```

### 4. `_format_example()`：模板格式化的核心

核心数据格式是：

```python
messages = [
    ["你好", "你好，请问有什么可以帮助你的？"],
    ["我发烧了", "请先测量体温。"],
]
```

方法首先选择 system prompt：

```python
system_prompt = system_prompt or self.system_prompt
```

- 调用者传入非空 system prompt：使用传入值；
- 未传入或传入空字符串：使用模板默认值。

所以传入 `system_prompt=""` 不能关闭一个非空的默认 system prompt，因为空字符串会回退到 `self.system_prompt`。

然后选择消息并遍历：

```python
messages = messages or self.messages

for turn_idx, [user_query, bot_resp] in enumerate(messages):
    if turn_idx == 0:
        convs.append(system_prompt + self.prompt.format(query=user_query))
        convs.append(bot_resp)
    else:
        convs.append(self.sep + self.prompt.format(query=user_query))
        convs.append(bot_resp)
```

其中：

- `self.prompt.format(query=user_query)` 把 `{query}` 替换为真实问题；
- 只有第一轮添加 system prompt；
- 后续轮次只在开头添加 `sep`；
- 每条消息必须正好包含 `[user_query, bot_resp]` 两项，否则列表解包会报错；
- 返回的是输入和回答交替出现的列表。

返回值近似为：

```python
[
    "system + 第一轮user格式",
    "第一轮assistant回答",
    "sep + 第二轮user格式",
    "第二轮assistant回答",
]
```

也就是：

```text
[source_0, target_0, source_1, target_1, ...]
```

### 5. `get_dialog()` 和 `get_prompt()`

二者都先调用 `_format_example()`，区别是是否把片段拼起来。

#### `get_dialog()`：保留分段

```python
def get_dialog(...):
    return self._format_example(messages, system_prompt)
```

返回：

```python
[source_0, target_0, source_1, target_1]
```

它主要用于 SFT。训练代码可以分别编码 source 和 target，并决定哪些 token 计算损失。

#### `get_prompt()`：拼成完整字符串

```python
def get_prompt(...):
    return "".join(self._format_example(messages, system_prompt))
```

它通过 `"".join()` 把所有片段合并成一个完整 prompt，主要用于 DPO 的 prompt 构造，也可以用于需要完整提示文本的场景。

如果最后一轮写成：

```python
["发烧怎么办？", ""]
```

拼接结果会结束在 assistant 开始标记后，等待模型生成回答。

### 6. 模板的注册与查找

所有手写模板保存在全局字典中：

```python
conv_templates: Dict[str, Conversation] = {}
```

导入 `training.template` 时，文件中的 `register_conv_template(...)` 会依次执行：

```python
def register_conv_template(template):
    conv_templates[template.name] = template
```

结果近似为：

```python
{
    "qwen": Conversation(...),
    "qwen3": Conversation(...),
    "qwen3_nothink": Conversation(...),
    "qwen3_5": Conversation(...),
    "qwen3_5_nothink": Conversation(...),
}
```

文件末尾通过名称查找：

```python
def get_conv_template(name: str) -> Conversation:
    return conv_templates[name]
```

注意：

- 名称不存在会产生 `KeyError`；
- 相同名称被再次注册时会覆盖旧模板；
- 返回的是字典中的原对象，不是副本，修改其 `messages` 会影响这个共享对象。

### 7. Qwen、Qwen3 和 Qwen3.5 模板

这些模板都采用相同的基本 ChatML 结构：

```python
system_prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
)

prompt = (
    "<|im_start|>user\n"
    "{query}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

sep = "\n"
stop_str = "<|im_end|>"
```

一次待生成的对话会被格式化为：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>

<|im_start|>user
发烧怎么办？<|im_end|>
<|im_start|>assistant
```

模型从最后的 assistant 标记后继续生成，并应在回答末尾生成 `<|im_end|>`。

当前文件中 `qwen3` 与 `qwen3_nothink` 的模板字段相同，`qwen3_5` 与 `qwen3_5_nothink` 也相同。仅选择名称带 `nothink` 的模板并没有在这段实现里自动加入 `/no_think`，也没有调用 `enable_thinking=False`。是否关闭思考模式还取决于最终 prompt、tokenizer 模板以及模型调用参数。

### 8. SFT 训练中的实际数据流

`supervised_finetuning.py` 中的处理过程是：

```text
原始 JSONL conversations
        ↓
识别 system / human / user / gpt / assistant 等角色
        ↓
统一成 user 和 assistant 消息
        ↓
组成 [[user, assistant], ...]
        ↓
指定 --template_name？
   ┌───────────────┴────────────────┐
   │ 是                              │ 否
   ↓                                 ↓
get_conv_template()          tokenizer.apply_chat_template()
   ↓                                 ↓
get_dialog()                 构造相同的 source/target 片段
   └───────────────┬────────────────┘
                   ↓
        分别编码 source 和 target
                   ↓
          拼接 input_ids 和 labels
```

训练代码逐轮执行：

```python
source_ids = tokenizer.encode(source, ...)
target_ids = tokenizer.encode(target, add_special_tokens=False)

input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
```

如果 `train_on_inputs=False`，标签近似为：

```text
input_ids: source tokens + target tokens + EOS
labels:    -100 ... -100 + target tokens + EOS
```

`-100` 表示这些 source token 不参与交叉熵损失计算，因此模型主要学习 assistant 应该生成的回答和回答结束位置。

### 9. DPO 与推理中的数据流

DPO 需要把历史消息和最后一个用户问题构造成完整 prompt：

```text
历史对话 + 最后一个 user 问题
               ↓
手写模板：get_prompt()
或 tokenizer.apply_chat_template(..., add_generation_prompt=True)
               ↓
             prompt
          ┌────┴────┐
          ↓         ↓
       chosen    rejected
```

当前 `demo/inference.py` 不直接调用 `training/template.py`，而是构造标准消息字典后调用：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

`add_generation_prompt=True` 通常会在末尾添加“现在轮到 assistant 回答”的控制标记。随后文本才被 tokenizer 转换为 `input_ids` 并交给 `model.generate()`。

### 10. BOS、EOS、`stop_str` 和 `sep`

| 名称         | 作用                                                 |
| ------------ | ---------------------------------------------------- |
| BOS          | Beginning Of Sequence，表示 token 序列开始           |
| EOS          | End Of Sequence，表示回答或序列结束                  |
| `stop_str` | 解码后的文本停止/截断字符串，例如 `<                 |
| `sep`      | 模板拼接时的轮次分隔符，例如换行，不一定具有停止作用 |

EOS 通常通过 `eos_token_id` 在 token 层控制 `generate()` 停止；`stop_str` 是字符串，常用于解码后的检查或清理。二者可能表示相同语义，但作用阶段不同。

特殊 token 配置错误可能导致：

- 缺少或未识别 EOS：回答结束后继续生成，直到 `max_new_tokens`；
- EOS 设置错误：正确结束符已经出现，但生成器没有停止；
- EOS 添加过早：回答提前结束；
- BOS/EOS 重复添加：输入格式偏离训练数据，可能出现特殊标记、重复或异常输出；
- assistant 开始标记缺失：模型可能继续模仿 user，而不是进入回答角色。

还要注意：`_format_example()` 只追加 `bot_resp`，不会自动在每个回答后拼接 `stop_str`。SFT 代码会在每轮 target 后追加 `tokenizer.eos_token_id`；如果在其他位置用 `get_prompt()` 构造多轮历史，则要检查历史回答之间是否具有正确的结束标记。

### 11. 为什么训练模板和推理模板必须一致

“一致”不要求训练和推理调用同一个 Python 函数，而是要求最终生成的文本格式和 token 序列遵守同一套协议。

例如训练时模型看到：

```text
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
回答<|im_end|>
```

推理时也应该用相同的角色标记和消息边界。如果推理时随意改成：

```text
问题：……
回答：
```

输入就偏离了模型训练时学到的格式，可能导致回答质量下降、角色混乱或不能正常停止。

当前项目可能出现两条路径：

- 训练指定 `--template_name qwen`：使用项目手写模板；
- 推理：使用 tokenizer 自带的 `chat_template`。

即使两边都声称采用 ChatML，也应打印最终文本或 token ID，核对角色标记、换行、BOS/EOS 和 assistant 开始标记是否完全一致。

### 12. 新模型优先使用 tokenizer 自带 `chat_template`

新模型通常在 tokenizer 配置中携带官方模板：

```python
tokenizer.chat_template
```

推荐将消息保持为标准结构：

```python
messages = [
    {"role": "system", "content": "你是一名医疗助手"},
    {"role": "user", "content": "发烧怎么办？"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

这样模板可以随模型和 tokenizer 一起维护，更容易正确处理特殊 token、思考模式、工具调用和模型版本差异。手写 `template.py` 更适合兼容旧模型、复现旧训练格式，或者 tokenizer 没有提供可用 chat template 的情况。

如果先执行：

```python
tokenizer.apply_chat_template(messages, tokenize=False)
```

再单独调用 tokenizer，要注意不要重复加入 BOS/EOS。直接令 `apply_chat_template(..., tokenize=True)` 通常更容易避免特殊 token 被重复添加，但当前项目为了分开 source 和 target、设置不同 labels，SFT 中采用了先生成文本片段再分别编码的方式。

### 13. 推荐的检查方法

先查看 tokenizer 自带模板的结果：

```python
messages = [
    {"role": "system", "content": "你是一名医疗助手"},
    {"role": "user", "content": "发烧怎么办？"},
]

print(tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
))
print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
print("有无 chat_template:", tokenizer.chat_template is not None)
```

再查看项目手写模板：

```python
from training.template import get_conv_template

template = get_conv_template("qwen")
history = [
    ["你好", "你好！"],
    ["发烧怎么办？", ""],
]

print(template.get_dialog(history))
print(template.get_prompt(history))
print(template.stop_str)
```

比较两边实际输出，而不是只比较模板名称，是确认训练和推理格式一致的最可靠方法。

### 14. 核心记忆

```text
Conversation
    └── 描述一种模型的完整对话协议

_format_example()
    └── 把多轮 [问题, 回答] 转成 source/target 交替片段

get_dialog()
    └── 保留片段，供 SFT 分别构造 input_ids 和 labels

get_prompt()
    └── 拼成完整字符串，供 DPO 或其他完整 prompt 场景使用

register_conv_template()
    └── 在模块导入时把模板注册到全局字典

get_conv_template()
    └── 按名称返回已经注册的模板
```

最重要的原则：

> 模型不仅学习回答内容，也学习角色、换行和特殊 token 的排列方式。训练和推理必须遵守同一套对话协议；对于新模型，应优先使用 tokenizer 自带的 `chat_template`。

## System prompt：训练、流式推理与批量推理

### 1. System prompt 是什么

System prompt 是对当前对话的高层约束，通常规定模型的身份、任务边界、输出格式和行为要求，例如：

```text
你是一个研究用医疗助手。请提供一般健康信息；信息不足时先说明不确定性，不进行远程确诊或提供个体处方剂量。
```

它与普通用户问题的角色不同：

```text
system
    └── 规定模型在整段对话中如何行动

user
    └── 提出本轮具体问题

assistant
    └── 在 system 和历史上下文约束下生成回答
```

System prompt 不是模型外部永久生效的设置。真正推理时，它会和其他消息一起被 chat template 格式化、tokenize，并成为当前样本 `input_ids` 的一部分。

### 2. 官方 chat template 接收的标准 `messages`

Qwen 等模型的 tokenizer 自带模板通常接收由字典组成的列表：

```python
messages = [
    {"role": "system", "content": "你是一个医疗助手。"},
    {"role": "user", "content": "我头疼。"},
    {"role": "assistant", "content": "请问头疼多久了？"},
    {"role": "user", "content": "大约两天。"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

历史对话没有单独的 `history` 角色，而是展开成按时间排列的多条 `user/assistant` 消息。最后一条是当前 `user` 问题，因为对应回答还没有生成。

推理时通常不需要手动加入：

```python
{"role": "assistant", "content": ""}
```

`add_generation_prompt=True` 会按照模型自己的协议，在结尾添加“现在轮到 assistant 回答”的起始标记。例如，对某些模板可能是 assistant 的角色头，但具体字符串由模型模板决定。

### 3. 流式多轮推理中的 system prompt

当前 `demo/inference.py` 使用以下内部结构保存历史：

```python
history = [
    [user_message_1, assistant_message_1],
    [user_message_2, assistant_message_2],
]
```

这个问题—答案对列表只是程序内部方便保存历史的格式。调用官方模板前，它会被转换成标准消息列表：

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message_1},
    {"role": "assistant", "content": assistant_message_1},
    {"role": "user", "content": user_message_2},
    {"role": "assistant", "content": assistant_message_2},
    {"role": "user", "content": current_query},
]
```

因此一条多轮推理样本实际包含：

```text
system prompt
+ 历史 user/assistant 消息
+ 当前 user 问题
+ assistant 生成起始标记
```

“流式”只表示生成出的 token 会被逐步解码和显示，不决定输入是否包含历史。流式推理可以是单轮，非流式推理也可以是多轮。

### 4. 批量推理中的 system prompt

当前 `batch_generate_answer()` 接收多个独立问题：

```python
sentences = [
    "什么是高血压？",
    "什么是糖尿病？",
    "用一句话介绍北京。",
]
```

程序为每个问题分别构造消息：

```text
样本1 = system prompt + 问题1
样本2 = system prompt + 问题2
样本3 = system prompt + 问题3
```

也就是说，同一个 `system_prompt` 字符串会被复用到多个样本，但最终会被分别编码进每个样本：

```python
for sentence in sentences:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]
```

它不是在 batch 外只输入一次，也不会在不同样本之间共享上下文。循环主要用于构造每条 prompt；同一个 batch 最终通过一次 `model.generate()` 并行生成多个独立回答。

### 5. 官方 Qwen 模板与 MedicalGPT `Conversation` 的区别

需要区分两种接口。

Qwen 等官方 tokenizer 模板使用：

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]

tokenizer.apply_chat_template(messages, ...)
```

MedicalGPT 的自定义 `Conversation` 常使用问题—答案对，并将 system prompt 单独传入：

```python
history_messages = [
    ["问题1", "回答1"],
    ["问题2", ""],
]

template.get_prompt(
    messages=history_messages,
    system_prompt="你是一个医疗助手。",
)
```

在这种自定义接口中，最后一个空回答可以表示该问题正在等待生成。但不能由此推断官方 Qwen 模板也接收问题—答案对；官方模板仍应使用 `role/content` 字典列表，并优先通过 `add_generation_prompt=True` 表示等待生成。

二者的目的相同，都是将结构化对话变为模型训练时认识的文本协议：

```text
官方 tokenizer 路径：
list[{role, content}, ...]
→ tokenizer.apply_chat_template()

MedicalGPT 自定义模板路径：
list[[问题, 回答], ...] + 单独 system_prompt
→ template.get_dialog()/get_prompt()
```

### 6. SFT 数据是否必须把 system 写进 `conversations`

不一定。当前 `data/sft/medical_sft_1K_format.jsonl` 的实际样本只有：

```json
{
  "conversations": [
    {"from": "human", "value": "用户问题"},
    {"from": "gpt", "value": "训练回答"}
  ]
}
```

这只能说明当前数据文件没有显式 system，不能推导出“所有 SFT 数据都不能包含 system”。本项目训练代码支持多种来源：

1. system 消息直接写在 `conversations` 中：

```json
{
  "conversations": [
    {"from": "system", "value": "你是一个医疗助手。"},
    {"from": "human", "value": "什么是高血压？"},
    {"from": "gpt", "value": "高血压是……"}
  ]
}
```

2. 使用每条记录的顶层 `system_prompt` 字段：

```json
{
  "system_prompt": "你是一个医疗助手。",
  "conversations": [
    {"from": "human", "value": "什么是高血压？"},
    {"from": "gpt", "value": "高血压是……"}
  ]
}
```

3. 数据只存问答，由预处理或自定义模板加入统一的默认 system prompt。

4. 数据和模板都不提供 system，模型只根据 user/assistant 对话进行训练。

同一个 system prompt 可以服务很多训练样本。存储上既可以在每条记录里重复，也可以在预处理时统一添加；但从模型看到的最终 token 序列来看，它仍需要实际出现在每条相应样本的上下文中。统一添加时必须记录 system prompt 的文本和版本，保证实验可复现。

### 7. 训练和推理的一致性

理想的 SFT 训练输入为：

```text
<system>任务和安全规则</system>
<user>用户问题</user>
<assistant>训练答案</assistant>
```

对应的推理输入为：

```text
<system>相同任务和安全规则</system>
<user>用户问题</user>
<assistant>
```

差别在于：

- 训练时 assistant 回答已经存在，回答 token 用于计算 loss；
- 推理时回答尚不存在，由模型继续生成；
- `add_generation_prompt=True` 添加 assistant 回答的开始位置。

如果训练时没有 system、输出是自由文本，推理时却加入复杂 system 并要求严格 JSON，基础模型可能依靠原有指令能力部分完成，但不能认为 SFT 已经学习了这套 system 规则。比较训练和推理是否一致时，应比较应用模板后的真实文本/token，而不只是比较变量名。

### 8. 核心记忆

```text
system prompt
    └── 是每条模型输入的上下文，不是模型外部永久生效的全局设置

流式多轮推理
    └── system + 历史问答 + 当前问题；流式只改变输出显示方式

批量推理
    └── 同一个 system 分别复制到每个独立样本，再组成 batch

官方 Qwen chat template
    └── 接收 role/content 字典列表

MedicalGPT Conversation
    └── 可接收问题—答案对，并将 system_prompt 单独传入

add_generation_prompt=True
    └── 添加 assistant 开始生成的角色标记，通常无需手写空 assistant 消息

SFT 数据
    └── 可以显式包含 system，也可以由预处理/模板统一添加
```

最准确的一句话是：

> System prompt 可以作为任务配置被多个样本复用，但模型处理每个样本时，它仍须成为该样本输入上下文的一部分；数据如何存储 system 可以不同，关键是训练和推理最终使用相同的角色顺序、模板协议和 system 语义。

## Tokenizer、BPE 与特殊 token

### 1. Tokenizer 的完整作用

大语言模型不能直接处理字符串，它真正处理的是整数形式的 token ID。Tokenizer 负责在人类文本与 token ID 之间转换：

```text
原始文本
   ↓ tokenize：按照分词规则切分
token字符串列表
   ↓ convert_tokens_to_ids
token ID列表
   ↓ 模型
logits / 生成的token ID
   ↓ decode
可读文本
```

在 `demo/test.py` 中：

```python
text = "老人突然一侧肢体无力、说话不清"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print(tokens)
print(ids)
print(tokenizer.decode(ids))
```

几个常用方法的区别：

| 方法 | 主要作用 | 返回值 |
| --- | --- | --- |
| `tokenizer.tokenize(text)` | 观察 tokenizer 内部如何切分文本 | token 字符串列表 |
| `tokenizer.convert_tokens_to_ids(tokens)` | 将已有 token 转为编号 | token ID 列表 |
| `tokenizer.encode(text)` | 分词并转成 token ID | token ID 列表 |
| `tokenizer(text)` | 为模型构造完整输入 | 包含 `input_ids`、`attention_mask` 等的字典 |
| `tokenizer.decode(ids)` | 将 token ID 还原为文本 | 字符串 |
| `tokenizer.convert_tokens_to_string(tokens)` | 将 token 字符串重新组合为可读文本 | 字符串 |

`encode()` 可以近似理解为：

```python
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
```

但实际实现还可能根据参数处理 BOS、EOS 等特殊 token，因此两种写法不保证在所有 tokenizer 上绝对等价。要明确控制特殊 token，可以使用：

```python
ids = tokenizer.encode(text, add_special_tokens=False)
```

### 2. `tokenize()` 不会自动应用聊天模板

下面这句只处理原始文本：

```python
tokenizer.tokenize(text)
```

它不会自动添加：

```text
<|im_start|>user
<|im_end|>
<|im_start|>assistant
```

要观察完整对话经过模板后的内容，应先构造标准消息：

```python
messages = [
    {"role": "system", "content": "你是一个研究用医疗助手。"},
    {"role": "user", "content": text},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

然后再分词：

```python
tokens = tokenizer.tokenize(prompt)
```

也可以让 chat template 直接返回 token ID：

```python
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
```

### 3. 为什么中文 token 看起来像乱码

当前 Qwen tokenizer 对以下文本执行：

```python
text = "老人突然一侧肢体无力、说话不清"
print(tokenizer.tokenize(text))
```

得到类似：

```python
[
    'èĢģäºº',
    'çªģçĦ¶',
    'ä¸Ģä¾§',
    'èĤ¢ä½ĵ',
    'æĹłåĬĽ',
    'ãĢģ',
    'è¯´è¯Ŀ',
    'ä¸įæ¸ħ',
]
```

这不是终端或中文文件编码损坏，而是字节级 BPE 的内部表示。处理过程可以简化为：

```text
中文字符
  ↓ 编码为UTF-8字节
UTF-8字节序列
  ↓ byte-to-unicode可逆映射
看似乱码的内部字符
  ↓ BPE合并
内部token
```

这些内部 token 大致对应：

| 内部 token | 原始文本 |
| --- | --- |
| `èĢģäºº` | 老人 |
| `çªģçĦ¶` | 突然 |
| `ä¸Ģä¾§` | 一侧 |
| `èĤ¢ä½ĵ` | 肢体 |
| `æĹłåĬĽ` | 无力 |
| `ãĢģ` | 、 |
| `è¯´è¯Ŀ` | 说话 |
| `ä¸įæ¸ħ` | 不清 |

字节级 tokenizer 的好处是任何 Unicode 文本最终都能转换成 UTF-8 字节，因此可以覆盖中文、英文、生僻字、Emoji、代码和特殊符号，减少无法表示的未知 token。

将内部 token 重新组合后，文本会恢复正常：

```python
tokens = tokenizer.tokenize(text)
print(tokenizer.convert_tokens_to_string(tokens))

ids = tokenizer.encode(text, add_special_tokens=False)
print(tokenizer.decode(ids))
```

预期都能恢复为：

```text
老人突然一侧肢体无力、说话不清
```

因此，`tokenize()` 更适合观察内部切分；面向用户展示结果时，应使用 `decode()` 或 `convert_tokens_to_string()`。

### 4. BPE 不要求特殊 token 位于词表开头

需要区分两种“位置”：

```text
序列位置：token在一句输入中的先后顺序
词表ID：token在整个词表中的唯一编号
```

例如：

```text
文本序列：<BOS>  我  喜欢  医学  <EOS>
序列位置：   0    1    2    3      4
token ID： 248045 105  892  3102 248046
```

BOS 位于输入序列的第一个位置，不代表 BOS 的词表 ID 必须是 0。BPE 算法也没有规定 BOS、EOS、PAD 必须使用词表最前面的编号。

不同 tokenizer 可能采用不同设计：

```text
设计一：0=<unk>，1=<s>，2=</s>，之后是普通词元
设计二：先建立普通基础词表，再把控制token追加到词表末尾
```

两种方式都可以。关键是训练和推理时，同一个 token 必须始终对应同一个 ID。

### 5. 基础词表不等于“完全没有特殊 token”

“是否属于基础词表”和“是否具有特殊语义”是两个不同维度：

| 所在位置 | 是否特殊 | 可能的例子 |
| --- | --- | --- |
| 基础词表 | 是 | 预先保留的 `<unk>`、`<s>`、`</s>` |
| 基础词表 | 否 | 普通文字、子词和字节 token |
| AddedToken | 是 | `<|im_start|>`、`<|im_end|>`、图像标记 |
| AddedToken | 否 | 需要整体识别但不希望自动隐藏的 `<think>` 等 token |

所以不能简单地认为：

```text
基础词表 = 普通token
AddedToken = 特殊token
```

有些 tokenizer 在训练基础词表时就预留特殊 token；有些则在基础词表训练完成后再追加聊天、多模态或工具调用所需的控制 token。

### 6. `vocab_size`、`len(tokenizer)` 与模型词表容量

终端打印：

```text
vocab_size=248044
```

表示当前 tokenizer 报告的基础词表有 248044 个 token。因为 ID 从 0 开始，基础 ID 范围是：

```text
0 ～ 248043
```

所以下一个 AddedToken 可以从 ID 248044 开始：

```text
248044 → <|endoftext|>
248045 → <|im_start|>
248046 → <|im_end|>
```

这没有编号冲突。

在 Hugging Face tokenizer 中，应区分：

```python
tokenizer.vocab_size
```

通常表示基础词表大小；而：

```python
len(tokenizer)
```

通常表示当前 tokenizer 实际可使用的全部 token 数量，即：

```text
基础词表 + AddedToken
```

还有第三个概念：

```python
model.config.vocab_size
```

它表示模型词嵌入层和输出层的词表容量。模型必须为 tokenizer 可能产生的 ID 准备对应的 embedding 行，因此通常应满足：

```text
模型embedding容量 >= tokenizer实际可用token数量
```

项目中的检查逻辑使用：

```python
model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)

if model_vocab_size != tokenizer_vocab_size:
    base_model.resize_token_embeddings(tokenizer_vocab_size)
```

这里比较 `len(tokenizer)` 而不是 `tokenizer.vocab_size`，就是为了把 AddedToken 计算进去。官方完整模型一般已经让模型 embedding 与 tokenizer 匹配；只有修改或扩展 tokenizer 后，才可能需要调整 embedding。

### 7. 为什么还需要 `AddedToken`

如果 `<|im_end|>` 没有被注册成一个完整 token，普通分词规则可能将其拆成多个片段：

```text
< + | + im + _ + end + | + >
```

这样它就无法稳定承担“消息结束”的控制语义。注册为 AddedToken 后：

```text
<|im_end|> → 一个完整token → 一个固定ID
```

AddedToken 的主要价值包括：

- 保证控制字符串作为一个整体，不经过普通 BPE 拆分；
- 在不重新训练整个 tokenizer 的情况下增加新 token；
- 保持已有普通 token 的 ID 不变；
- 为聊天、多模态、工具调用、代码补全和思考标记扩展词表。

终端中的 `AddedToken(...)` 是 tokenizer 配置恢复出来的对象，不表示当前 `inference.py` 刚刚动态添加了它。模型发布者已经把这些 token 保存到 tokenizer 配置中，`from_pretrained()` 加载时会自动还原。

### 8. `special_tokens` 与 `added_tokens_decoder` 为什么会重复显示

它们是在从不同角度描述同一个 token。

`special_tokens` 描述功能身份：

```text
eos_token → <|im_end|>
pad_token → <|endoftext|>
```

`added_tokens_decoder` 描述 ID、字符串和处理规则：

```text
248046 → AddedToken("<|im_end|>", special=True, ...)
```

所以同一个 `<|im_end|>` 同时出现，并不代表创建了两份：

```text
功能视角：它是EOS
词表视角：它的ID是248046，并具有AddedToken的切分规则
```

`AddedToken` 与 `special=True` 也不是同一个概念：

```text
AddedToken(..., special=True)
    → 扩展token，并注册为特殊token

AddedToken(..., special=False)
    → 扩展token，但解码时可以作为普通内容显示
```

例如 `skip_special_tokens=True` 通常会隐藏 `<|im_end|>`，但不一定隐藏 `special=False` 的 `<think>`。

### 9. 推荐的 tokenizer 观察代码

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    trust_remote_code=True,
)

text = "老人突然一侧肢体无力、说话不清"

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text, add_special_tokens=False)

print("内部token：", tokens)
print("token ID：", ids)
print("恢复文本：", tokenizer.decode(ids))
print("直接组合：", tokenizer.convert_tokens_to_string(tokens))

print("基础词表大小：", tokenizer.vocab_size)
print("实际token总数：", len(tokenizer))
print("EOS：", tokenizer.eos_token, tokenizer.eos_token_id)
print("<|im_end|> ID：", tokenizer.convert_tokens_to_ids("<|im_end|>"))
```

### 10. 分词部分核心记忆

```text
tokenize()
    └── 展示内部token字符串，不一定适合人类直接阅读

encode()
    └── 将文本转换成模型使用的token ID

decode()
    └── 将token ID恢复成人类可读文本

字节级BPE
    └── 奇怪字符是UTF-8字节的可逆内部映射，不是文件乱码

tokenizer.vocab_size
    └── 通常是基础词表大小

len(tokenizer)
    └── 通常是基础词表加AddedToken后的实际总量

AddedToken
    └── 保证新增字符串作为一个完整token，并保持原词表ID稳定
```

最容易混淆但必须记住的一句话：

> BOS 位于输入序列开头，不代表它的 token ID 位于词表开头；特殊 token 既可能属于基础词表，也可能通过 AddedToken 加入。
