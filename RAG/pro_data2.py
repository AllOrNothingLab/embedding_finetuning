"""
python pro_data2.py \
  --input_file qa_results.json \
  --train_val_split 0.2 \
  --output_path qa_rag_dataset.json
"""
import os
import json
import argparse
import random
from collections import defaultdict
from tqdm import tqdm
from evaldataset import RAGDataset  # 确保 RAGDataset 类可用

# 设置随机种子保证结果可复现
random.seed(42)

def load_and_process_data(input_file):
    """加载并处理原始JSON数据，构建三列数据集"""
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 按键排序确保相邻关系
    sorted_keys = sorted(raw_data.keys())
    processed_data = []
    
    for idx, key in enumerate(sorted_keys):
        qa_pairs = raw_data[key]
        group_size = len(qa_pairs)
        
        for i, pair in enumerate(qa_pairs):
            query = pair["question"]
            positive = pair["answer"]
            
            # 负样本选择策略
            if group_size > 1:
                # 同组其他回答作为负样本
                other_answers = [p["answer"] for j, p in enumerate(qa_pairs) if j != i]
                negative = random.choice(other_answers)
            else:
                # 相邻组回答作为负样本
                adjacent_key = None
                if idx > 0:  # 优先选择前一组
                    adjacent_key = sorted_keys[idx-1]
                elif idx < len(sorted_keys)-1:  # 次选后一组
                    adjacent_key = sorted_keys[idx+1]
                
                if adjacent_key and raw_data.get(adjacent_key):
                    negative = random.choice([p["answer"] for p in raw_data[adjacent_key]])
                else:
                    # 无相邻组可用时的回退方案
                    negative = positive  # 应尽量避免此情况
            
            processed_data.append({
                "Query": query,
                "Positive Document": positive,
                "Hard Negative Document": negative
            })
    
    return processed_data

def main(args: argparse.Namespace):
    # 1. 加载并处理原始数据
    processed_data = load_and_process_data(args.input_file)
    
    # 2. 转换为 RAGDataset 格式
    rag_dataset = RAGDataset()
    corpus_id_map = {}
    
    for idx, item in enumerate(tqdm(processed_data)):
        query = item["Query"]
        positive = item["Positive Document"]
        negative = item["Hard Negative Document"]
        
        # 创建唯一查询ID
        query_id = f"q-{idx}"
        rag_dataset.queries[query_id] = query
        rag_dataset.relevant_docs[query_id] = []
        rag_dataset.negative_docs[query_id] = []
        
        # 处理正文档
        if positive not in corpus_id_map:
            corpus_id = f"c-{len(corpus_id_map)}"
            corpus_id_map[positive] = corpus_id
            rag_dataset.corpus[corpus_id] = positive
        rag_dataset.relevant_docs[query_id].append(corpus_id_map[positive])
        
        # 处理负文档
        if negative not in corpus_id_map:
            corpus_id = f"c-{len(corpus_id_map)}"
            corpus_id_map[negative] = corpus_id
            rag_dataset.corpus[corpus_id] = negative
        rag_dataset.negative_docs[query_id].append(corpus_id_map[negative])
    
    # 3. 拆分数据集
    rag_dataset.split(args.train_val_split, seed=42)
    
    # 4. 保存结果
    print(f"Corpus size: {len(rag_dataset.corpus)}")
    print(f"Train queries: {len(rag_dataset.queries_split['train'])}")
    print(f"Validation queries: {len(rag_dataset.queries_split['val'])}")
    rag_dataset.save(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str, required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--train_val_split",
        type=float, default=0.2,
        help="Train/validation split ratio"
    )
    parser.add_argument(
        "--output_path",
        type=str, required=True,
        help="Output file path"
    )
    args = parser.parse_args()
    
    main(args)