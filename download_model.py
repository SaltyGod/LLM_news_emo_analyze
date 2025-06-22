from modelscope import snapshot_download
import os

# 指定本地目录
os.environ["MODELSCOPE_CACHE"]="/root/.cache/LLMS/hub"

# 下载模型到指定目录
# model_dir1 = snapshot_download('BAAI/bge-large-zh-v1.5')
# model_dir2 = snapshot_download('BAAI/bge-reranker-v2-m3')
# model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')

print(f"模型已下载到: {model_dir}")
