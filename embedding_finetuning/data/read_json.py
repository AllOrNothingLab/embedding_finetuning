import json

# 读取 JSON 文件
with open('/workspace/volume/gxs2/zht/project4/embedding_finetuning/data/infgrad_retrieval_data_llm.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有顶级键名（包括 "queries"）
top_level_keys = list(data.keys())
print("所有顶级键名:", top_level_keys)

# 仅提取 "queries" 的同级键名（排除 "queries" 自身）
sibling_keys = [key for key in data.keys() if key != "queries"]
print("queries的同级键名（排除自身）:", sibling_keys)

# ['queries', 'corpus', 'relevant_docs', 'negative_docs', 'reference_answers', 'queries_split']