import os
import torch
import logging
import pandas as pd
import numpy as np
import gc
from typing import List, Dict, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

os.environ["USE_FLASH_ATTENTION"] = "1"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型默认路径
DEFAULT_MODEL_PATH = "/root/onethingai-fs/mymodels/0524_full_para_model/checkpoint-7200"

class QwenInferenceEngine:
    """Qwen模型推理引擎，使用vLLM进行高效推理"""
    
    def __init__(self, model_path=None, config=None):
        # 优化默认配置
        self.config = {
            "max_new_tokens": 1024,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1,
            "batch_size": 48,              
            "max_input_length": 2500,      # 减小输入长度
            "clear_cuda_cache": True,
            "dynamic_batch_size": True,
            "min_batch_size": 32,          
            "cuda_empty_freq": 10,         
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.75,
            "prefetch_factor": 2,          
            "num_workers": 4               
        }
        
        if config:
            self.config.update(config)
            
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH)
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            self._clear_cuda_cache()
            
            if torch.cuda.is_available():
                logger.info(f"检测到{torch.cuda.device_count()}个GPU设备")
                self.device = "cuda"
                torch.cuda.set_device(0)
                torch.cuda.stream(torch.cuda.Stream())
            else:
                logger.warning("未检测到GPU，将使用CPU进行推理")
                self.device = "cpu"
            
            # 修改vLLM引擎配置
            self.model = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=self.config["tensor_parallel_size"],
                dtype="bfloat16",
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.75),
                block_size=32,             # 减小block_size
                swap_space=1,              
                max_num_seqs=128,          
                enforce_eager=True,        
                max_num_batched_tokens=32768, # 增加到与模型最大长度相同
                max_model_len=32768         # 显式设置模型最大长度
            )
            
            # 验证模型是否正常工作
            # self._validate_model()
            
            logger.info(f"成功加载Qwen模型，设备:{self.device}")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            self._clear_cuda_cache()  # 失败时也清理缓存
            raise RuntimeError(f"模型初始化失败: {str(e)}")
    
    def _validate_model(self):
        """验证模型是否正常加载和工作"""
        try:
            # 简单测试
            test_input = "Hello"
            
            # 创建采样参数
            sampling_params = SamplingParams(
                max_tokens=5,
                temperature=0.4,
                top_p=0.9
            )
            
            # 执行简单推理测试
            self.model.generate([test_input], sampling_params)
            logger.info("模型验证通过")
            
            # 清理测试产生的缓存
            self._clear_cuda_cache()
            
        except Exception as e:
            logger.error(f"模型验证失败: {str(e)}")
            self._clear_cuda_cache()
            raise RuntimeError(f"模型验证失败: {str(e)}")
    
    def _clear_cuda_cache(self):
        """清理CUDA缓存释放内存"""
        if not torch.cuda.is_available() or not self.config["clear_cuda_cache"]:
            return
            
        # 确保所有GPU操作完成
        torch.cuda.synchronize()
        
        # 手动触发垃圾回收
        gc.collect()
        
        # 清空CUDA缓存
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            
        # 检查并报告当前内存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                logger.debug(f"GPU:{i} 分配内存:{mem_allocated:.2f}GB, 保留内存:{mem_reserved:.2f}GB")
    
    def generate_response(self, prompt, **kwargs):
        """单条输入的推理函数，保持向后兼容性"""
        results = self.generate_batch([prompt], **kwargs)
        return results[0] if results else None
    
    def generate_batch(self, prompts, **kwargs):
        """批量处理多个输入的推理函数"""
        if not prompts:
            logger.warning("收到空输入列表")
            return []
        
        # 合并配置参数，允许临时覆盖默认配置
        config = self.config.copy()
        if kwargs:
            config.update(kwargs)
        
        results = []
        batch_size = config["batch_size"]
        batch_counter = 0
        
        # 分批处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # 处理批次
            try:
                batch_results = self._process_single_batch(batch_prompts, config)
                results.extend(batch_results)
                
                # 增加计数器
                batch_counter += 1
                
                # 定期清理内存
                if config["clear_cuda_cache"] and batch_counter % config["cuda_empty_freq"] == 0:
                    logger.debug(f"清理CUDA缓存（第{batch_counter}批次后）")
                    self._clear_cuda_cache()
                    
            except RuntimeError as e:
                # 处理内存不足错误
                if "CUDA out of memory" in str(e) and config["dynamic_batch_size"] and batch_size > config["min_batch_size"]:
                    # 清理内存
                    self._clear_cuda_cache()
                    
                    # 减小批处理大小
                    new_batch_size = max(batch_size // 2, config["min_batch_size"])
                    logger.warning(f"CUDA内存不足，减小批处理大小: {batch_size} -> {new_batch_size}")
                    batch_size = new_batch_size
                    config["batch_size"] = new_batch_size
                    
                    # 重试当前批次
                    i -= batch_size  # 回退以重新处理当前批次
                    continue
                else:
                    # 其他错误，清理内存并重新抛出
                    self._clear_cuda_cache()
                    logger.error(f"批处理错误: {str(e)}")
                    # 返回对应数量的None
                    batch_results = [None] * len(batch_prompts)
                    results.extend(batch_results)
            
        # 完成所有批次后清理内存
        self._clear_cuda_cache()
        return results
    
    def _process_single_batch(self, batch_prompts, config):
        """处理单批次输入"""
        try:
            # 使用异步数据处理
            formatted_prompts = []
            for prompt in batch_prompts:
                if len(prompt) > config["max_input_length"]:
                    prompt = prompt[:config["max_input_length"]]
                formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                formatted_prompts.append(formatted_prompt)
            
            sampling_params = SamplingParams(
                max_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                repetition_penalty=config["repetition_penalty"],
                stop=["<|im_end|>"]
            )
            
            # 使用CUDA事件来同步
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            outputs = self.model.generate(
                formatted_prompts,
                sampling_params,
                use_tqdm=False
            )
            end_event.record()
            
            # 等待生成完成并计算时间
            end_event.synchronize()
            generation_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
            
            # 记录性能指标
            logger.debug(f"批处理耗时: {generation_time:.2f}秒, "
                        f"每样本平均时间: {generation_time/len(batch_prompts):.3f}秒")
            
            batch_responses = []
            for output in outputs:
                response_text = output.outputs[0].text.strip()
                if "<|im_end|>" in response_text:
                    response_text = response_text.split("<|im_end|>")[0].strip()
                batch_responses.append(response_text)
            
            return batch_responses
            
        except Exception as e:
            logger.error(f"批处理错误: {str(e)}")
            self._clear_cuda_cache()
            raise
        
def process_csv_file(csv_path, output_path=None, input_column="qwen_input", output_column="qwen_output", **kwargs):
    """
    批量处理CSV文件中的输入，只处理output_column为空的行
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出文件路径，默认为原文件名_processed.csv
        input_column: 输入列名
        output_column: 输出列名
        **kwargs: 可选的配置参数，会覆盖默认配置
    """
    try:
        # 设置默认输出路径
        if output_path is None:
            base_name = os.path.splitext(csv_path)[0]
            output_path = f"{base_name}_infered.csv"
        
        # 读取CSV文件
        logger.info(f"正在读取CSV文件: {csv_path}")
        # df = pd.read_csv(csv_path,encoding = 'utf_8_sig',lineterminator='\n')
        df = pd.read_excel(csv_path)
        df['qwen_score'] = None
        df['qwen_output'] = None
        
        # 检查输入列是否存在
        if input_column not in df.columns:
            raise ValueError(f"输入列 '{input_column}' 在CSV文件中不存在")
        
        # 确保输出列存在
        if output_column not in df.columns:
            df[output_column] = None
        
        # 判断哪些行需要处理（output_column为空的行）
        empty_mask = df[output_column].isna() | (df[output_column] == "") | (df[output_column].astype(str) == "nan")
        rows_to_process = df[empty_mask].index.tolist()
        
        total_rows = len(df)
        empty_rows = len(rows_to_process)
        
        logger.info(f"CSV文件共有 {total_rows} 行，其中 {empty_rows} 行需要处理")
        
        # 如果没有需要处理的行，直接返回
        if empty_rows == 0:
            logger.info("没有空值行需要处理，直接返回原文件")
            return df
        
        # 只提取需要处理的行的输入
        inputs_to_process = df.loc[rows_to_process, input_column].fillna("").tolist()
        
        # 设置更加保守的默认配置以避免OOM
        default_config = {
            "batch_size": 32,               # vLLM支持更大批次
            "clear_cuda_cache": True,       # 启用内存清理
            "dynamic_batch_size": True,     # 启用动态批大小
            "cuda_empty_freq": 5,           # 降低清理频率
            "tensor_parallel_size": 1       # 单GPU模式
        }
        
        # 合并用户配置
        config = default_config.copy()
        config.update(kwargs)
        
        # 初始化模型引擎
        engine = QwenInferenceEngine(config=config)
        batch_size = engine.config["batch_size"]
        
        logger.info(f"开始处理 {empty_rows} 行空值数据，初始批次大小: {batch_size}")
        
        # 批量处理数据并显示进度
        results_map = {}  # 使用字典存储结果，键为行索引
        processed_count = 0
        
        # 尝试从临时文件恢复部分结果
        if os.path.exists(f"{output_path}.temp"):
            try:
                temp_df = pd.read_csv(f"{output_path}.temp")
                if output_column in temp_df.columns:
                    # 只复制已处理的行
                    for idx in rows_to_process:
                        if idx < len(temp_df) and not pd.isna(temp_df.loc[idx, output_column]) and temp_df.loc[idx, output_column] != "":
                            results_map[idx] = temp_df.loc[idx, output_column]
                            processed_count += 1
                    
                    if processed_count > 0:
                        logger.info(f"从临时文件恢复了 {processed_count}/{empty_rows} 行结果")
            except Exception as temp_e:
                logger.warning(f"读取临时文件失败: {str(temp_e)}")
        
        # 更新需要处理的行
        rows_to_process = [idx for idx in rows_to_process if idx not in results_map]
        inputs_to_process = df.loc[rows_to_process, input_column].fillna("").tolist()
        
        logger.info(f"还需处理 {len(rows_to_process)} 行数据")
        
        try:
            # 按批次处理
            for i in tqdm(range(0, len(rows_to_process), batch_size), desc="批处理进度"):
                batch_indices = rows_to_process[i:i+batch_size]
                batch_inputs = inputs_to_process[i:i+batch_size]
                
                # 处理一批数据
                batch_results = engine.generate_batch(batch_inputs)
                
                # 将结果与原始行索引对应起来
                for j, result in enumerate(batch_results):
                    idx = batch_indices[j]
                    results_map[idx] = result
                
                processed_count += len(batch_indices)
                
                # 定期保存中间结果
                if i > 0 and i % 10000 == 0:
                    # 将已处理的结果更新到DataFrame
                    temp_df = df.copy()
                    for idx, result in results_map.items():
                        temp_df.loc[idx, output_column] = result
                    
                    # 保存临时文件
                    temp_df.to_csv(f"{output_path}.temp", index=False,encoding = 'utf_8_sig')
                    logger.info(f"已保存中间结果 ({processed_count}/{empty_rows} 行)")
        
        except KeyboardInterrupt:
            logger.warning("用户中断处理，保存当前进度")
            
        finally:
            # 将所有结果更新到原始DataFrame
            for idx, result in results_map.items():
                df.loc[idx, output_column] = result
            
            # 保存最终结果
            df.to_csv(output_path, index=False)
            logger.info(f"处理完成，共处理 {len(results_map)}/{empty_rows} 行，结果已保存至: {output_path}")
            
            # 删除临时文件
            if os.path.exists(f"{output_path}.temp"):
                os.remove(f"{output_path}.temp")
            
            # 清理资源
            engine._clear_cuda_cache()
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return df
        
    except Exception as e:
        logger.error(f"处理CSV文件失败: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

# 向后兼容的全局函数
_global_engine = None

def generate_response(prompt, max_length=512, temperature=0.5):
    """兼容原始API的推理函数"""
    global _global_engine
    if _global_engine is None:
        _global_engine = QwenInferenceEngine()
    return _global_engine.generate_response(prompt, max_new_tokens=max_length, temperature=temperature)

# 使用示例
if __name__ == "__main__":
    csv_path = "/root/onethingai-fs/mydata/process_data/Macro_report_70393_rag_processed.xlsx"
    
    # 为vLLM优化的参数
    config = {
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "batch_size": 64,              # 调整batch_size
        "clear_cuda_cache": True,      
        "dynamic_batch_size": True,
        "min_batch_size": 32,
        "cuda_empty_freq": 10,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.75,
        "prefetch_factor": 2,
        "num_workers": 4
    }
    
    # 设置CUDA后端优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    processed_df = process_csv_file(
        csv_path=csv_path,
        input_column="qwen_input",
        output_column="qwen_output",
        **config
    )
    
    # 输出处理统计
    empty_after = processed_df["qwen_output"].isna().sum()
    filled = len(processed_df) - empty_after
    print(f"处理完成: 共 {len(processed_df)} 行，已填充 {filled} 行，剩余空值 {empty_after} 行")