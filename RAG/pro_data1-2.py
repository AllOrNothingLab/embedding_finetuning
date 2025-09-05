import json
import re
import os
import time
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 设置环境变量
os.environ["MODELSCOPE_CACHE"] = "/workspace/volume/gxs2/zht/Models"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/volume/gxs2/zht/Models"
os.environ["HF_HOME"] = "/workspace/volume/gxs2/zht/Models"

def calculate_question_count(text_length):
    """根据文本长度确定问题数量"""
    if text_length < 100:
        return 1
    elif text_length < 300:
        return 2
    elif text_length < 600:
        return 3
    elif text_length < 1000:
        return 4
    else:
        return 5

def extract_answers_from_text(questions, text):
    """从原文中抽取答案片段（RAG风格）"""
    answers = []
    for question in questions:
        # 构建提示要求模型返回原文片段
        prompt = (
            f"请根据以下问题，从提供的文本中找出最相关的连续片段作为答案。"
            f"答案必须是文本中的原话，不要添加任何解释或修改。\n"
            f"问题：{question}\n"
            f"文本：{text}\n"
            f"答案："
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # 生成模型输入
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        # 调用模型生成答案
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,  # 限制生成长度，确保是片段
            temperature=0.1,     # 低温度确保确定性
            top_p=0.9,
            do_sample=False
        )
        
        # 解析输出
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            # 尝试提取思考内容
            index = len(output_ids) - output_ids[::-1].index(151668)
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        # 验证答案是否来自原文
        if content not in text:
            # 如果不在原文中，尝试找到最匹配的片段
            start_idx = text.find(content[:10]) if len(content) > 10 else -1
            if start_idx != -1 and len(content) > 10:
                end_idx = min(start_idx + len(content) + 20, len(text))
                content = text[start_idx:end_idx].split('。')[0] + '。'  # 截取到句号
            else:
                # 使用启发式方法找到相关片段
                sentences = re.split(r'(?<=[。！？])', text)
                best_match = max(sentences, key=lambda s: len(set(question) & set(s)), default="")
                content = best_match if best_match else text[:100]  # 默认取前100字符
        
        answers.append(content)
        time.sleep(1)  # 避免频繁调用
    
    return answers

def main():
    global tokenizer, model
    
    # 加载模型和tokenizer
    model_name = "Qwen/Qwen3-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 读取原始JSON文件
    with open("test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 准备结果字典
    results = {}
    
    # 按顺序处理每个片段 (00-17)
    for key in sorted(data.keys()):
        text = data[key]
        print(f"\nProcessing segment {key}...")
        
        # 跳过空片段
        if not text or text == "[empty]":
            print(f"Skipping empty segment {key}")
            results[key] = []
            continue
            
        # 计算问题数量
        num_questions = calculate_question_count(len(text))
        print(f"Text length: {len(text)}, Generating {num_questions} questions")
        
        # 第一阶段：生成问题
        prompt = (
            f"你是一个专业的问题生成助手。请根据以下文本片段生成{num_questions}个相关问题。"
            f"问题应该覆盖文本的主要内容和关键细节。\n"
            f"输出格式要求：\n"
            f"Q1: [问题1]\n"
            f"Q2: [问题2]\n"
            f"...\n"
            f"注意：不要生成答案，只生成问题。不要添加任何额外解释或格式标记。\n\n"
            f"文本片段：\n{text}"
        )
        
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        # 调用模型生成问题
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        
        # 解析输出
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        # 解析生成的问题
        questions = []
        for line in content.split('\n'):
            if line.startswith('Q') and ':' in line:
                question = line.split(':', 1)[1].strip()
                questions.append(question)
        
        # 确保问题数量匹配
        questions = questions[:num_questions]
        if len(questions) < num_questions:
            print(f"Warning: Only generated {len(questions)} questions, expected {num_questions}")
        
        # 第二阶段：从原文中抽取答案
        answers = extract_answers_from_text(questions, text)
        
        # 组合QA对
        qa_pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        
        # 存储结果
        results[key] = qa_pairs
        
        # 打印当前结果
        print(f"Generated {len(qa_pairs)} QA pairs:")
        for i, pair in enumerate(qa_pairs):
            print(f"  Q{i+1}: {pair['question']}")
            print(f"  A{i+1}: {pair['answer'][:60]}...")
        
        # 添加延迟以避免过载
        time.sleep(5)
    
    # 保存结果到新JSON文件
    with open("qa_results2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nProcessing completed. Results saved to qa_results.json")

if __name__ == "__main__":
    main()