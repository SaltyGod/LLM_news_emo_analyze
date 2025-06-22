# 导入
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
# import wandb
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# # ✅ 初始化 wandb
# wandb.init(project="market-sentiment-finetune")

# ✅ 加载模型
max_seq_length = 2048
model_path = "/root/.cache/LLMS/hub/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
output_dir = "/root/LLM_news_emo_analyze/DeepSeek_output/DeepSeek-R1-NewsEmo"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print(f"正在加载模型: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 确保tokenizer有正确的pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ 添加特殊标记到tokenizer
special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
num_added_tokens = tokenizer.add_special_tokens(special_tokens)
print(f"添加了 {num_added_tokens} 个特殊标记到tokenizer")

# 调整模型以适应新的token
model.resize_token_embeddings(len(tokenizer))

# ✅ 加载数据
train_path = "/root/LLM_news_emo_analyze/DATA/deepseek_sft_0419data/ds_train_data.csv"
eval_path = "/root/LLM_news_emo_analyze/DATA/deepseek_sft_0419data/ds_valid_data.csv"

print(f"正在加载训练数据: {train_path}")
train_df = pd.read_csv(train_path, encoding='utf_8_sig', lineterminator='\n')
print(f"正在加载验证数据: {eval_path}")
eval_df = pd.read_csv(eval_path, encoding='utf_8_sig', lineterminator='\n')

# 数据清洗
train_df.dropna(subset=["Question", "Complex_CoT", "Response"], inplace=True)
eval_df.dropna(subset=["Question", "Complex_CoT", "Response"], inplace=True)

# ✅ 准备训练数据 - 修改数据准备方式
def prepare_data(df):
    result = []
    for _, row in df.iterrows():
        # 使用简单的指令格式 
        input_text = row['Question'].strip()
        output_text = f"<think>{row['Complex_CoT'].strip()}</think>\n{row['Response'].strip()}"
        
        # 创建包含指令格式的文本
        result.append({
            "text": f"指令: {input_text}\n\n输出: {output_text}"
        })
    return Dataset.from_pandas(pd.DataFrame(result))

print("正在准备训练和验证数据...")
train_dataset = prepare_data(train_df)
eval_dataset = prepare_data(eval_df)

# ✅ 应用LoRA
print("正在配置LoRA...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,  # 增加rank以提高性能
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,  # 添加少量dropout提高鲁棒性
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ 训练配置 - 改进训练参数
print("正在配置训练参数...")
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    num_train_epochs=5,  # 增加训练轮次
    warmup_ratio=0.1,  # 使用预热比例而不是固定步数
    learning_rate=2e-5,  # 降低学习率
    logging_steps=10,
    eval_steps=50,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # 使用余弦学习率调度
    seed=3407,
    output_dir="outputs",
    save_strategy="epoch",
    save_total_limit=4,
)

# 使用兼容的SFTTrainer配置
print("正在创建训练器...")

# 定义一个简单的格式化函数，返回文本
def formatting_func(example):
    return example["text"]

# 创建SFTTrainer，移除有问题的参数
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    args=training_args,
)

# ✅ 开始训练
print("开始训练...")
trainer.train()

# ✅ 保存模型
print(f"训练完成，正在保存模型到 {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("模型训练完成！")

# # 结束wandb会话
# wandb.finish()