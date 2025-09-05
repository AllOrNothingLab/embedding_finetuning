# 让LLM生成问题Q和回答A
import json
import re
import os
import time
import torch
import torch_mlu
from torch_mlu.utils.model_transfer import transfer
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

def parse_qa_output(output):
    """解析模型输出的问题-答案对"""
    qa_pairs = []
    # 使用正则表达式匹配Q/A对
    pattern = r'Q\d+: (.*?)\nA\d+: (.*?)(?=\nQ|\Z)'
    matches = re.findall(pattern, output, re.DOTALL)
    
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()
        qa_pairs.append({"question": question, "answer": answer})
    
    # 如果正则匹配失败，尝试简单分割
    if not qa_pairs:
        lines = output.split('\n')
        for i in range(0, len(lines), 2):
            if i+1 < len(lines) and lines[i].startswith('Q') and lines[i+1].startswith('A'):
                q = lines[i].split(':', 1)[1].strip()
                a = lines[i+1].split(':', 1)[1].strip()
                qa_pairs.append({"question": q, "answer": a})
    
    return qa_pairs

def main():
    # 加载模型和tokenizer
    model_name = "Qwen/Qwen3-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 读取原始JSON文件
    with open("test2.json", "r", encoding="utf-8") as f:
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
        
        # 构建提示
        prompt = (
            f"你是一个专业的问题生成助手。请根据以下文本片段生成{num_questions}个相关问题，"
            f"并为每个问题提供准确的答案。问题和答案必须严格基于文本内容。\n"
            f"输出格式要求：\n"
            f"Q1: [问题1]\n"
            f"A1: [答案1]\n"
            f"Q2: [问题2]\n"
            f"A2: [答案2]\n"
            f"...\n"
            f"注意：不要添加任何额外解释或格式标记。\n\n"
            f"文本片段：\n{text}"
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
        
        # 调用模型生成
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9
        )
        
        # 解析输出
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            # 尝试提取思考内容
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        print("Generated content:")
        print(content)
        
        # 解析问题-答案对
        qa_pairs = parse_qa_output(content)
        print(f"Parsed {len(qa_pairs)} QA pairs")
        
        # 存储结果
        results[key] = qa_pairs
        
        # 添加延迟以避免过载
        time.sleep(5)
    
    # 保存结果到新JSON文件
    with open("qa_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nProcessing completed. Results saved to qa_results.json")

if __name__ == "__main__":
    main()