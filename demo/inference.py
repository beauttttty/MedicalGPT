# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Inference script

python3 demo/inference.py --base_model Qwen/Qwen3.5-2B --lora_model outputs-sft-qwen-v1 --interactive
"""
import argparse#解析模型参数
import json#把推理结果写成json
import os#检查删除文件
from threading import Thread#创建线程，使模型生成与流式读取能够同时进行。

import torch
from peft import PeftModel#加载lora模型等peft适配器
from tqdm import tqdm#显示推理进度条
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,#自动加载模型和分词器
    TextIteratorStreamer,#用于流式生成文本，逐段返回模型生成的文本
    GenerationConfig,#加载模型自带的生成参数
    BitsAndBytesConfig,#配置8bit/4bit量化参数
)

#流式生成函数
@torch.inference_mode()#这是一个装饰器，效果近似于在整个函数外包一层：with torch.inference_mode():，把函数当作参数传递
#表示在推理阶段不需要计算梯度，节省显存和计算资源，@ 是装饰器的语法糖。所谓语法糖，是指一种更简洁、更容易阅读的写法，并没有创造完全不同的底层能力
def stream_generate_answer(
        model,
        tokenizer,
        prompt,#套好对话模板的输入文字
        device,#输入张量要放的设备，例如GPU
        do_print=True,#后四个有默认值，不一定传入，是否边生成边打印
        max_new_tokens=512,#生成的最大token数
        temperature=0.7,#采样温度，控制生成文本的随机性，值越大，生成的文本越随机，值越小，生成的文本越确定
        repetition_penalty=1.0,#重复惩罚系数，控制生成文本中重复内容的惩罚程度，值越大，模型越不容易生成重复内容
        stop_str="</s>",#生成文本的停止标记，模型生成到这个标记就会停止生成
):
    """Generate answer from prompt with GPT and stream the output，根据prompt 流式生成回答"""
    # Streamer for handling token generation
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    #tokenizer:解码，timeout=60.0：等待新输出的超时时间为 60 秒，skip_prompt=True：不把输入 prompt 再输出一遍；skip_special_tokens=True，隐藏 BOS、EOS 等特殊 token
    inputs = tokenizer(prompt, return_tensors="pt")#对输入的prompt进行编码，返回pytorch张量，input_id以及attention_mask
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)#attention_mask是一个二进制张量，用于指示模型在处理输入序列时应该关注哪些位置。它的形状与输入序列相同，通常包含0和1，其中1表示该位置的token是有效的，模型应该关注，而0表示该位置是填充的，模型应该忽略。
    generation_kwargs = dict(#创建参数字典
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,#true表示使用采样生成文本，false表示使用贪心搜索生成文本（确定性生成）
        repetition_penalty=repetition_penalty,
        streamer=streamer,#把生成结果交给streamer处理，streamer会把生成的token解码成文本并返回
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)#Thread类创建一个新线程，target指定线程要执行的函数，这里是model.generate，kwargs指定传递给函数的关键字参数，这里是generation_kwargs字典
    thread.start()#使用线程目的：因为 model.generate() 在生成结束前通常不会返回。将它放到后台线程，主线程就能同时迭代 streamer，实现“生成一点、显示一点”。
    #**字典 表示把字典展开成关键字参数，比如 dict(a=1, b=2) 会被展开成 a=1, b=2
    generated_text = ""#用于累计每次从 streamer 取得的文字片段。一个完整回答分成好几段，treamer 中的所有文本片段共同组成这一个 prompt 对应的一条回答
    for new_text in streamer:
        stop = False
        pos = new_text.find(stop_str)#查找停止标记，找不到时返回-1
        if pos != -1:
            new_text = new_text[:pos]#截取停止标记前的文本
            stop = True
        generated_text += new_text
        if do_print:
            print(new_text, end="", flush=True)#flush=True表示立即刷新输出缓冲区，确保文本实时显示在终端上，而不是等待缓冲区满了才显示
        if stop:
            break#遇到停止标记则停止生成
    if do_print:
        print()
    return generated_text

#批量生成函数
@torch.inference_mode()
def batch_generate_answer(
        sentences,#一个问题列表，例如["介绍下北京", "乙肝和丙肝的区别？"]
        model,
        tokenizer,
        system_prompt,#统一添加给所有问题的系统提示，例如"你是一个有帮助的助手。"
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
):
    """Generate answers from prompts in batch mode"""
    generated_texts = []#用于存储每个问题的生成回答
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
    )
#创建批量推理的公共生成参数。这里没有 streamer，因为批量模式等待整个 batch 生成结束后再统一解码
    # Construct batch prompts
    prompts = []#为问题构建完整的prompt列表，包含系统提示和用户问题
    for s in sentences:
        # Construct dialogue message format
        messages = [{"role": "user", "content": s}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Generate prompt using apply_chat_template
        prompt = tokenizer.apply_chat_template(#应用对话模板，将系统提示和用户问题组合成完整的输入prompt
            messages,
            tokenize=False,#不进行tokenize，先返回字符串形式的prompt，后续统一tokenize
            add_generation_prompt=True,#在结尾加上“现在轮到 assistant 回答”的标记
        )
        prompts.append(prompt)

    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(device)
    attention_mask = inputs_tokens['attention_mask'].to(device)
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)#进行批量生成，[batch_size, 输入长度 + 生成长度]，它既包含原始 prompt，也包含模型生成的新 token。
    for gen_sequence in outputs:#其是批量处理，不只一个问题，每个问题的生成结果都在 outputs 中
        prompt_len = len(input_ids[0])#第一条输入序列长度
        gen_sequence = gen_sequence[prompt_len:]#截取生成的部分，去掉原始 prompt，只保留模型生成的新 token
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=True)#把tokenid转回人类可读文字，skip_special_tokens=True代表。
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        gen_text = gen_text.strip()#去掉字符串开头和结尾的空格、换行等空白字符。
        generated_texts.append(gen_text)

    return generated_texts


def main():
    parser = argparse.ArgumentParser()#创建·ArgumentParser对象，用于解析命令行参数
    parser.add_argument('--base_model', default=None, type=str, required=True)#定义一个命令行参数，参数名：--base_model，类型：字符串
    #指定基础模型的路径或名称，例如--base_model Qwen/Qwen3.5-2B
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--system_prompt', default="", type=str, help="System prompt")
    parser.add_argument('--stop_str', default="", type=str)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)#1.0 表示不施加重复惩罚，大于 1.0 会抑制重复
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,#问题文件
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive mode (default multi-turn)")#action='store_true' 的含义是：当用户在命令行中使用 --interactive 参数时，该参数的值会被设置为 True。
    parser.add_argument('--single_tune', action='store_true', help='Whether to use single-turn model')#action='store_true' 的含义是：当用户在命令行中使用 --single_tune 参数时，该参数的值会被设置为 True。
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8-bit mode')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4-bit mode')
    args = parser.parse_args()#parse_args() 读取终端命令，返回一个命名空间对象，args.base_model
    print(args)
    load_type = 'auto'#让 Transformers 根据模型文件自动判断权重类型
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    # AutoTokenizer 会根据模型名称或路径自动选择合适的分词器类，并加载相应的预训练分词器权重和配置。padding_side='left'表示在输入序列的左侧进行填充，使得所有输入序列具有相同的长度。
    #from_pretrained() 接受 Hugging Face 模型 ID 或本地目录；trust_remote_code=True 允许执行模型仓库提供的自定义 Python 代码
    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_type,
        "low_cpu_mem_usage": True,
        "device_map": 'auto',
    }
    #量化配置
    if args.load_in_8bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_type,
        )
    #加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **config_kwargs)
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, using default.")
    #这里尝试从模型目录加载 generation_config.json。如果找不到文件并产生 OSError，就保留默认生成配置
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)#获取基础模型的词汇表大小，即嵌入层的权重矩阵的第一维度大小
        tokenzier_vocab_size = len(tokenizer)#获取分词器的词汇表大小，即分词器中可识别的唯一 token 的数量
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resizing model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)#调整模型的嵌入层大小以适应分词器的词汇表大小

    if args.lora_model:#加载lora模型
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')#基础模型 + LoRA 适配器 → model
        print("Loaded LoRA model")
    else:
        model = base_model
    model.eval()#设置模型为评估模式，禁用 dropout 等训练特定行为，它的作用是确保模型在推理阶段的行为与训练阶段一致，从而提高推理的稳定性和准确性。
#等价于model.train(False)，不会进行dropout等训练特定行为，确保推理阶段的行为与训练阶段一致，提高推理的稳定性和准确性。
    print(tokenizer)#它的语法是：print(tokenizer) 会调用 tokenizer 对象的 __str__ 方法，返回一个字符串，描述分词器的类型、词汇表大小、特殊 token 等信息。它的作用是帮助用户了解当前使用的分词器的基本情况。
    # Test data
    if args.data_file is None:
        examples = ["介绍下北京", "乙肝和丙肝的区别？"]
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]#strip() 删除每行首尾的空格和换行。
        print("First 10 examples:")
        for example in examples[:10]:
            print(example)

    # Chat
    system_prompt = args.system_prompt
    stop_str = args.stop_str or "</s>"  # Use default stop string

    if args.interactive:
        print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
        history = []#存储历史对话，格式为 [[user_msg, assistant_msg], ...]，每个元素是一个列表，包含用户消息和助手消息
        while True:#无限循环，直到用户输入 exit
            try:#读取输入和异常处理
                query = input(f"User: ")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please try again.")
                continue
            except Exception:
                raise#Raise的作用是将捕获到的异常重新抛出，让上层调用者处理。它通常用于在异常处理块中捕获异常后，决定不处理该异常，而是将其传递给上层调用者，以便进行进一步的处理或记录日志。
            if query == "":
                print("Please input text, try again.")
                continue
            if query.strip() == "exit":#strio\p() 方法用于移除字符串首尾的空白字符（包括空格、制表符、换行符等），并返回一个新的字符串。它不会修改原始字符串，而是返回一个新的字符串副本。
                print("Exiting...")
                break
            if query.strip() == "clear":
                history = []
                print("History cleared.")
                continue

            print(f"Assistant: ", end="", flush=True)#flush=True 表示立即刷新输出缓冲区，确保文本实时显示在终端上，而不是等待缓冲区满了才显示
            if args.single_tune:
                history = []

            # Construct dialogue messages
            messages = []#构建完整消息列表
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})#系统提示

            # Add historical dialogue
            for user_msg, assistant_msg in history:#遍历历史对话，添加到消息列表中
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Add current user query
            messages.append({"role": "user", "content": query})#加入当前用户问题

            # Generate prompt using apply_chat_template
            prompt = tokenizer.apply_chat_template(#应用对话模板，将系统提示、历史对话和当前用户问题组合成完整的输入prompt
                messages,
                tokenize=False,
                add_generation_prompt=True,#在结尾加上“现在轮到 assistant 回答”的标记
            )

            response = stream_generate_answer(#流式生成，prompt 传入的是完整的对话上下文，模型会根据上下文生成回答
                model,
                tokenizer,
                prompt,
                model.device,
                do_print=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            if history:
                history[-1][-1] = response.strip()
            history.append([query, response.strip()])
    else:
        print("Starting inference.")#批量推理
        counts = 0
        if os.path.exists(args.output_file):
            os.remove(args.output_file)#删除已存在的输出文件，确保每次推理结果都是新的
        eval_batch_size = args.eval_batch_size#批量推理的批次大小
        for batch in tqdm(#tqdm 是一个用于显示进度条的库，它可以在长时间运行的循环中显示当前进度，给用户提供反馈。这里用于显示批量推理的进度。
                [
                    examples[i: i + eval_batch_size]#列表推导式
                    for i in range(0, len(examples), eval_batch_size)
                ],
                desc="Generating outputs",
        ):
            responses = batch_generate_answer(
                batch,#问题列表，例如 ["介绍下北京", "乙肝和丙肝的区别？"]
                model,
                tokenizer,
                system_prompt,
                model.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            results = []
            for example, response in zip(batch, responses):
                print(f"===")
                print(f"Input: {example}")
                print(f"Output: {response}\n")
                results.append({"Input": example, "Output": response})
                counts += 1
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for entry in results:
                    json.dump(entry, f, ensure_ascii=False)#写入json文件，ensure_ascii=False表示不转义非ASCII字符，保持原始字符编码
                    f.write('\n')#每条 JSON 后写一个换行，所以输出是 JSONL
                   # 示例：
                    #{"Input": "什么是乙肝？", "Output": "乙肝是由乙型肝炎病毒引起的……"}
                    #{"Input": "什么是发烧？", "Output": "发烧通常指体温高于正常范围……"}
        print(f'Saved to {args.output_file}, size: {counts}')


if __name__ == '__main__':
    main()
