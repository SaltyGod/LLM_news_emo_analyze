from modelscope import snapshot_download
import os

# 指定本地目录
os.environ["MODELSCOPE_CACHE"]="./root/.cache/LLMS"

# 下载模型到指定目录
model_dir = snapshot_download('BAAI/bge-large-zh-v1.5')

print(f"模型已下载到: {model_dir}")
