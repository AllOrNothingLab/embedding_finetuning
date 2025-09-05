import sys
sys.path.append("..")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置镜像站
os.environ["WANDB_DISABLED"] = "true"  # 完全禁用Wandb
import torch
import torch_mlu 
from torch_mlu.utils.model_transfer import transfer
import datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# 1. Load a model to finetune 
model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5"
)

from evaldataset import RAGDataset
dataset = RAGDataset.from_file("qa_rag_dataset.json")
train_dataset, dataset_keys = dataset.get_train_dataset(split="train", negative_num=1, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
eval_dataset, _ = dataset.get_train_dataset(split="val", negative_num=1, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")

# for i in train_dataset[:2]:
#     print(i)


loss = MultipleNegativesRankingLoss(model)
# args = SentenceTransformerTrainingArguments(
#     # Required parameter:
#     output_dir="./checkpoint/bge-small-zh-v1.5-sft",
#     num_train_epochs=1,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     gradient_accumulation_steps=16, # global batch size = 32 * 16 = 512
#     learning_rate=2e-5,
#     warmup_ratio=0.1,
#     fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
#     bf16=False,  # Set to True if you have a GPU that supports BF16
#     batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
#     # Optional tracking/debugging parameters:
#     eval_strategy="steps",
#     eval_steps=20,
#     save_strategy="steps",
#     save_steps=1000,
#     save_total_limit=1,
#     logging_steps=20,
#     seed=42,
#     lr_scheduler_type="cosine",
#     optim="adamw_torch_fused",
# )
num_train_epochs = 30  # 增加epoch数补偿小数据集
per_device_batch_size = 8  # 减小batch size
# 计算合理步数
total_steps = (len(train_dataset) * num_train_epochs) / per_device_batch_size
eval_steps = max(10, int(total_steps * 0.05))  # 每5%步数评估一次
save_steps = max(50, int(total_steps * 0.2))   # 每20%步数保存一次
args = SentenceTransformerTrainingArguments(
    output_dir="./checkpoint/bge-small-zh-v1.5-sft",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,  # 小数据集不需要梯度累积
    learning_rate=5e-5,  # 小数据集可适当提高学习率
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=2,
    logging_steps=5,  # 更频繁的日志记录
    seed=42,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    report_to="none",  # 禁用所有报告工具
    logging_dir="./logs",
    disable_tqdm=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    max_steps=-1,  # 确保使用epoch而非步数限制
)

from typing import Any

# fix transformer not compatible with sentence_transformers
class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        return super().compute_loss(model, inputs, return_outputs)
       
    
trainer = CustomSentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=datasets.Dataset.from_list(train_dataset).select_columns(
        dataset_keys
    ),
    eval_dataset=datasets.Dataset.from_list(eval_dataset).select_columns(
        dataset_keys
    ),
    loss=loss,
)
trainer.train()

model.save("./checkpoint/bge-small-zh-v1.5-sft")
