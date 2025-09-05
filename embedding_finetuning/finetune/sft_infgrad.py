import sys
sys.path.append("..")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置镜像站
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

from eval.dataset import RAGDataset
dataset = RAGDataset.from_file("../data/infgrad_retrieval_data_llm.json")
train_dataset, dataset_keys = dataset.get_train_dataset(split="train", negative_num=1, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
eval_dataset, _ = dataset.get_train_dataset(split="val", negative_num=1, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")

# for i in train_dataset[:2]:
#     print(i)


loss = MultipleNegativesRankingLoss(model)
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="../checkpoint/bge-small-zh-v1.5-sft",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=16, # global batch size = 32 * 16 = 512
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=100000,
    save_total_limit=1,
    logging_steps=20,
    seed=42,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
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

model.save("../checkpoint/bge-small-zh-v1.5-sft")
