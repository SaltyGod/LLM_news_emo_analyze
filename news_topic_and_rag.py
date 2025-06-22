from transformers import AutoTokenizer, AutoModel
import torch
import re
import torch.nn.functional as F
import pandas as pd
import json
import faiss
import numpy as np
from tqdm import tqdm
import time
from modelscope import AutoModelForSequenceClassification, AutoTokenizer as MSTokenizer
from functools import lru_cache
from retry import retry
import os
from concurrent.futures import ProcessPoolExecutor
import math
import psutil

def load_model(model_path):
    """
    加载模型和分词器，使用FP16精度。

    参数:
    model_path (str): 模型的路径。

    返回:
    tokenizer: 加载的分词器。
    model: 加载的模型。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    # 转换为FP16精度
    model = model.half()  # 转换为FP16
    model.eval()
    return tokenizer, model

def get_embeddings(prompts, tokenizer, model):
    """
    计算给定提示的嵌入表示。

    参数:
    prompts (list): 自定义提示的列表。
    tokenizer: 加载的分词器。
    model: 加载的模型。

    返回:
    torch.Tensor: 嵌入表示的张量。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Tokenize custom prompts，设置最大长度为2048
    encoded_input = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt', max_length=2048)
    # 将输入移到GPU，但保持整数类型
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        # 模型已经是half精度，会自动处理精度转换
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.cpu().float()  # 转回FP32用于CPU操作

    return sentence_embeddings

def calculate_cosine_similarity(embedding_1, embeddings):
    """
    计算一个嵌入与其他多个嵌入的余弦相似度。

    参数:
    embedding_1 (torch.Tensor): 形状为 [1, hidden_size] 的张量
    embeddings (torch.Tensor): 形状为 [num_topics, hidden_size] 的张量

    返回:
    torch.Tensor: 相似度分数列表
    """
    # 确保维度正确
    if len(embedding_1.shape) == 2:
        embedding_1 = embedding_1.squeeze(0)
    if len(embeddings.shape) == 2:
        embeddings = embeddings

    # 计算余弦相似度
    cosine_similarities = F.cosine_similarity(embedding_1.unsqueeze(0), embeddings)
    return cosine_similarities

def get_theme_prompts(lda_key_words_path):
    """
    从Excel文件中读取主题和关键词，生成主题提示。
    
    参数:
    lda_key_words_path (str): 主题关键词Excel文件路径
    
    返回:
    list: 主题提示列表
    """
    try:
        df = pd.read_excel(lda_key_words_path)
        if df.empty:
            raise ValueError("主题关键词文件为空")
            
        prompts = []
        for index, row in df.iterrows():
            if pd.isna(row['主题类别']):
                continue
                
            theme_category = row['主题类别']
            # 过滤掉NaN值并转换为列表
            keywords = [str(k) for k in row.iloc[1:].dropna().tolist()]
            if keywords:
                prompt = f"新闻的主题是“{theme_category}”，关键词是“{''.join(keywords)}”"
                prompts.append(prompt)
        
        if not prompts:
            raise ValueError("未能生成有效的主题提示")
            
        return prompts
    except Exception as e:
        print(f"读取主题关键词文件出错: {str(e)}")
        raise

# 分隔句子
def split_sentences(text):
    # 使用正则表达式分隔句子
    sentences = re.split(r'[\n]', text)
    # 去掉空字符串
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def extract_theme_category(topic_text):
    """
    从主题文本中提取主题类别。
    
    参数:
    topic_text (str): 形如"新闻的主题是XX，关键词是YY"的文本
    
    返回:
    str: 主题类别
    """
    try:
        # 使用正则表达式提取主题类别
        match = re.search(r'主题是(.*?)，关键词是', topic_text)
        if match:
            return match.group(1)
        return topic_text  # 如果无法提取，返回原文本
    except Exception as e:
        print(f"提取主题类别时出错: {str(e)}")
        return topic_text

def load_reranker(reranker_path):
    """
    加载reranker模型，使用FP16精度
    """
    tokenizer = MSTokenizer.from_pretrained(reranker_path)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_path)
    
    # 转换为FP16精度
    model = model.half()  # 转换为FP16
    model.eval()
    return tokenizer, model

def compute_reranker_score(model, tokenizer, text1, text2):
    """
    使用reranker计算两段文本的相关性得分
    """
    with torch.no_grad():
        inputs = tokenizer([text1], [text2], padding=True, truncation=True, return_tensors='pt', max_length=2048)
        scores = model(**inputs).logits.view(-1,).float()
    return scores.item()

def hybrid_sentiment_search(sentence, faiss_index, sentiment_prompts, original_words, sentiment_scores,
                          reranker_model, reranker_tokenizer, bge_tokenizer, bge_model, 
                          first_stage_threshold=0.48, top_k=6):
    """
    两阶段搜索：FAISS快速筛选 + Reranker精确排序
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构造新闻句子的prompt
    sentence_prompt = f"新闻内容是“{sentence}”"
    
    # 第一阶段：FAISS快速筛选（直接使用已构建的索引）
    sentence_embedding = get_embeddings([sentence_prompt], bge_tokenizer, bge_model)
    D, I = faiss_index.search(sentence_embedding.numpy(), top_k)
    
    # 筛选出相似度大于阈值的候选词
    candidates = []
    for sim, idx in zip(D[0], I[0]):
        if sim > first_stage_threshold:
            candidates.append({
                'word': original_words[idx],
                'score': sentiment_scores[idx],
                'similarity': sim,
                'index': idx
            })
    
    if not candidates:
        return []
    
    # 第二阶段：Reranker精确排序
    reranker_results = []
    for candidate in candidates:
        # 构造文本对
        pairs = [[sentence_prompt, f"与新闻内容最相关的单词是“{candidate['word']}”"]]
        
        # 使用reranker计算分数
        with torch.no_grad():
            inputs = reranker_tokenizer(pairs, padding=True, truncation=True, 
                                      return_tensors='pt', max_length=2048)
            # 将输入移到GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            scores = reranker_model(**inputs, return_dict=True).logits.view(-1,).float()
            reranker_score = scores[0].item()
        
        reranker_results.append({
            'word': candidate['word'],
            'score': candidate['score'],
            'reranker_score': reranker_score
        })
    
    # 按reranker分数排序并返回前2个结果
    sorted_results = sorted(reranker_results, key=lambda x: x['reranker_score'], reverse=True)
    return sorted_results[:2]

def batch_get_embeddings(texts, tokenizer, model, batch_size=300):
    """
    批量计算向量，使用GPU加速
    """
    embeddings_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=2048)
        # 将输入移到GPU，但保持整数类型
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = sentence_embeddings.cpu().float()  # 转回FP32用于CPU操作
            embeddings_list.append(sentence_embeddings)
            
    return torch.cat(embeddings_list, dim=0)

@lru_cache(maxsize=300)  # 增加缓存大小
def get_cached_embeddings(text, tokenizer, model, is_batch=False):
    """
    带缓存的向量计算，增加缓存大小
    """
    if is_batch:
        return batch_get_embeddings(text, tokenizer, model)
    return batch_get_embeddings([text], tokenizer, model)

def is_valid_sentence(sentence):
    """
    检查句子是否有效
    
    参数:
    sentence (str): 待检查的句子
    
    返回:
    bool: 是否为有效句子
    """
    # 无效的关键词列表
    invalid_keywords = [
        "文章来源", "建议关注", "买入评级", "风险提示", "增持评级",
        "不构成任何投资建议", "亦不代表平台观点", "请投资人独立判断和决策", "图片来源"
    ]
    
    # 检查句子长度
    if len(sentence.strip()) < 110:
        return False
        
    # 检查是否包含无效关键词
    for keyword in invalid_keywords:
        if keyword in sentence:
            return False
            
    return True

@retry(tries=3, delay=1, backoff=2)
def process_single_sentence(sentence, faiss_index, sentiment_prompts, original_words, 
                          sentiment_scores, reranker_model, reranker_tokenizer, 
                          bge_tokenizer, bge_model, key_topic_embeddings, 
                          key_topic_list, similarity_threshold):
    """
    处理单个句子，带重试机制
    
    返回:
    dict/None: 如果句子有效且匹配成功返回结果字典，否则返回None
    """
    # 首先检查句子是否有效
    if not is_valid_sentence(sentence):
        return None
    
    # 情感词匹配
    sentiment_matches = hybrid_sentiment_search(
        sentence, faiss_index, sentiment_prompts, original_words, sentiment_scores,
        reranker_model, reranker_tokenizer, bge_tokenizer, bge_model
    )
    
    # 如果没有匹配到情感词，直接返回None
    if not sentiment_matches:
        return None
    
    # 主题匹配
    sentence_embedding = get_cached_embeddings(sentence, bge_tokenizer, bge_model)
    topic_similarity_scores = calculate_cosine_similarity(sentence_embedding, key_topic_embeddings)
    max_topic_idx = torch.argmax(topic_similarity_scores).item()
    topic_similarity = topic_similarity_scores[max_topic_idx].item()
    
    # 如果主题相似度低于阈值，返回None
    if topic_similarity <= similarity_threshold:
        return None
    
    # 构建结果字典
    result_dict = {
        "topic": extract_theme_category(key_topic_list[max_topic_idx]),
        "similarity": topic_similarity,
    }
    
    # 添加情感词匹配结果
    sentiment_prompt = f"新闻文本是“{sentence}”，包含的情绪词典是："
    sentiment_prompt += ",".join([f"{m['word']}:{m['score']}" for m in sentiment_matches])
    result_dict["prompt"] = sentiment_prompt
    
    return result_dict

def batch_process_sentences(sentences, faiss_index, sentiment_prompts, original_words, 
                          sentiment_scores, reranker_model, reranker_tokenizer, 
                          bge_tokenizer, bge_model, key_topic_embeddings, 
                          key_topic_list, similarity_threshold):
    """
    批量处理句子
    
    返回:
    list: 每个句子的处理结果，无效句子返回None
    """
    results = []
    
    # 批量计算句子嵌入（用于情感词搜索）
    print("计算批次句子嵌入...")
    sentence_prompts = [f"新闻内容是“{s}”" for s in sentences]
    sentence_embeddings = get_embeddings(sentence_prompts, bge_tokenizer, bge_model)
    print("句子嵌入计算完成")
    
    # 一次性处理所有句子
    print("开始处理句子...")
    with tqdm(total=len(sentences), desc="批内处理进度") as pbar:
        for i, sentence in enumerate(sentences):
            try:
                # 使用已计算的嵌入进行情感词匹配
                result = process_single_sentence(
                    sentence, faiss_index, sentiment_prompts, original_words,
                    sentiment_scores, reranker_model, reranker_tokenizer,
                    bge_tokenizer, bge_model, key_topic_embeddings,
                    key_topic_list, similarity_threshold
                )
                results.append(result)
                
            except Exception as e:
                print(f"处理句子出错: {str(e)}")
                results.append(None)
            
            pbar.update(1)
            
    return results

def process_news_data(news_file_path, lda_key_words_path, sentiment_dict_path, 
                     model_path, reranker_path, output_file_path, 
                     similarity_threshold=0.485,batch_size=300):
    try:
        # 加载BGE-large模型
        bge_tokenizer, bge_model = load_model(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bge_model = bge_model.to(device)
        
        # 加载reranker模型
        reranker_tokenizer, reranker_model = load_reranker(reranker_path)
        reranker_model = reranker_model.to(device)
        print("成功加载Reranker模型")
        
        # 加载情感词典
        sentiment_prompts, original_words, sentiment_scores = load_sentiment_words(sentiment_dict_path)
        print(f"成功加载 {len(sentiment_prompts)} 个情感词")
        
        # 批量计算情感词向量
        print("批量计算情感词向量...")
        sentiment_word_prompts = [f"与新闻内容最相关的单词是“{word}”" for word in original_words]
        sentiment_embeddings = batch_get_embeddings(sentiment_word_prompts, bge_tokenizer, bge_model, batch_size)
        sentiment_embeddings_np = sentiment_embeddings.numpy()
        
        # 构建FAISS索引
        print("开始构建FAISS索引...")
        faiss_index = build_faiss_index(sentiment_embeddings_np)
        print("FAISS索引构建完成")
        
        # 获取主题提示和嵌入
        key_topic_list = get_theme_prompts(lda_key_words_path)
        key_topic_embeddings = get_embeddings(key_topic_list, bge_tokenizer, bge_model)
        print(f"成功加载 {len(key_topic_list)} 个主题提示")
        
        # 处理新闻数据
        print("开始读取新闻数据...")
        df = pd.read_excel(news_file_path)

        if df.empty:
            raise ValueError("新闻文件为空")
        print(f"共读取 {len(df)} 条新闻数据")
        
        # 收集所有有效句子
        all_valid_sentences = []
        sentence_to_news_index = {}  # 记录句子来自哪篇新闻
        
        # 添加进度条：收集句子阶段
        print("\n开始收集有效句子...")
        with tqdm(total=len(df), desc="收集句子进度") as pbar:
            for index, row in df.iterrows():
                if pd.isna(row['NewsContent_process']):
                    pbar.update(1)
                    continue
                
                news_content = str(row['NewsContent_process'])
                sentences = split_sentences(news_content)
                
                for sentence in sentences:
                    if is_valid_sentence(sentence):
                        all_valid_sentences.append(sentence)
                        sentence_to_news_index[sentence] = index
                
                pbar.update(1)
        
        print(f"\n收集完成，共找到 {len(all_valid_sentences)} 个有效句子")
        
        # 批量处理所有句子
        batch_size = 300  # 可根据GPU内存调整
        results_dict = {}
        num_processed = 0
        
        # 添加进度条：批处理句子阶段
        print("\n开始批量处理句子...")
        with tqdm(total=len(all_valid_sentences), desc="处理句子进度") as pbar:
            for i in range(0, len(all_valid_sentences), batch_size):
                # 显示当前批次信息
                end_idx = min(i + batch_size, len(all_valid_sentences))
                current_batch_size = end_idx - i
                
                print(f"\n处理批次 {i//batch_size + 1}/{math.ceil(len(all_valid_sentences)/batch_size)}, "
                      f"当前批次大小: {current_batch_size}")
                
                batch = all_valid_sentences[i:end_idx]
                
                # 批量处理
                batch_results = batch_process_sentences(
                    batch, faiss_index, sentiment_prompts, original_words,
                    sentiment_scores, reranker_model, reranker_tokenizer,
                    bge_tokenizer, bge_model, key_topic_embeddings,
                    key_topic_list, similarity_threshold
                )
                
                # 收集结果
                valid_count = 0
                for sentence, result in zip(batch, batch_results):
                    if result:  # 如果结果有效
                        valid_count += 1
                        index = sentence_to_news_index[sentence]
                        if index not in results_dict:
                            results_dict[index] = {'valid_sentences': {}, 'topic_match_dict': {}}
                        
                        results_dict[index]['valid_sentences'][f"news_prompt{len(results_dict[index]['valid_sentences'])+1}"] = sentence
                        results_dict[index]['topic_match_dict'][sentence] = result
                
                num_processed += current_batch_size
                pbar.update(current_batch_size)
                
                # 每个批次结束后显示有效匹配率和内存使用情况
                print(f"批次有效匹配率: {valid_count}/{current_batch_size} ({valid_count/current_batch_size*100:.2f}%)")
                monitor_memory()
                
                # 显示总体进度
                print(f"总进度: {num_processed}/{len(all_valid_sentences)} ({num_processed/len(all_valid_sentences)*100:.2f}%)")
                print(f"当前已匹配新闻数: {len(results_dict)}")
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"\n处理完成，共处理 {len(results_dict)} 条有效数据")
        
        # 更新DataFrame
        print("\n开始更新数据...")
        with tqdm(total=len(results_dict), desc="更新进度") as pbar:
            for index, result in results_dict.items():
                df.at[index, 'NewsContent_cut'] = json.dumps(result['valid_sentences'], ensure_ascii=False)
                df.at[index, 'cut_news_topic_match'] = json.dumps(result['topic_match_dict'], ensure_ascii=False)
                pbar.update(1)
        
        # 保存结果
        print("\n开始保存结果...")
        df.to_excel(output_file_path, index=False)
        print(f"处理完成，结果已保存到 {output_file_path}")
        
        
        
############# ==========================  添加的额外处理 ==========================  #############

        # ✅ 设置要保留的原始列（可以按需修改）
        columns_to_keep = ['DeclareDate', 'Title','NewsContent', 'NewsSource', 'yearmonth']  # 举例：只保留这两列

        # ✅ 提取 JSON 中的字段
        def extract_json_fields(cell):
            try:
                if pd.isna(cell):
                    return pd.DataFrame()
                json_obj = json.loads(str(cell))
                result = []
                for context, value in json_obj.items():
                    result.append({
                        'context': context,
                        'topic': value.get('topic'),
                        'similarity': value.get('similarity'),
                        'prompt': value.get('prompt')
                    })
                return pd.DataFrame(result)
            except Exception as e:
                print(f"解析失败：{e}\n内容为：{cell}")
                return pd.DataFrame()

        # ✅ 从 prompt 中提取新闻文本和情绪词典
        def extract_from_prompt(prompt):
            try:
                match = re.search(r'新闻文本是“(.*?)”，包含的情绪词典是：(.*)', prompt)
                if match:
                    news_text = match.group(1)
                    sentiment_dict = match.group(2)
                    return pd.Series({'cut_news': news_text, 'sentiment_dict': sentiment_dict})
                else:
                    return pd.Series({'cut_news': None, 'sentiment_dict': None})
            except Exception as e:
                print(f"提取失败：{e}\n内容为：{prompt}")
                return pd.Series({'cut_news': None, 'sentiment_dict': None})

        # ✅ 主处理逻辑
        all_rows = []
        for i, row in df.iterrows():
            expanded = extract_json_fields(row['cut_news_topic_match'])
            if not expanded.empty:
                for col in df.columns:
                    if col != 'cut_news_topic_match':
                        expanded[col] = row[col]
                expanded['cut_news_topic_match'] = row['cut_news_topic_match']
                all_rows.append(expanded)

        if all_rows:
            final_df = pd.concat(all_rows, ignore_index=True)

            # 保留原始列 + 插入新列
            original_cols = df.columns.tolist()
            extracted_fields = final_df['prompt'].apply(extract_from_prompt)
            final_df = pd.concat([final_df, extracted_fields], axis=1)

            # 可选：只保留你想要的原始列 + 所有新列
            selected_cols = columns_to_keep + [col for col in final_df.columns if col not in df.columns]
            final_df = final_df[selected_cols]

            print("rag展开成功")
        else:
            print("❌ 没有成功解析任何 JSON 行")
        
        print('开始解析英文词典')
            
        def extract_bracket_content_and_value(text):
            """
            从文本中提取括号内的内容和后面的分值，支持负数、零和小数
            输入格式例如: "抑制因素/阻碍因素(disincentive):-0.3,药物/毒品(drug):0.0"
            输出格式例如: "disincentive:-0.3,drug:0.0"
            """
            # 修改正则表达式以匹配包括负数和零在内的所有数值
            pattern = r'\(([^)]+)\):([-]?[0-9]*\.?[0-9]+)'
            matches = re.findall(pattern, text)
            
            # 将匹配结果组合为需要的格式
            if matches:
                result = ','.join([f"{term}:{value}" for term, value in matches])
                return result
            else:
                return ""
        final_df['sentiment_dict_v2'] = final_df['sentiment_dict'].apply(lambda x: extract_bracket_content_and_value(x) if pd.notna(x) else x)
        final_df.drop('prompt', axis=1, inplace=True)
        print('英文词典解析成功')
        
        print('\n开始进行prompt构造')
        prompt2 = """**角色定义**
        你是一位擅长中英双语的中国市场情绪分析专家，能够基于任务流程对新闻文本的市场情绪进行客观、准确的评级：

        **任务流程**
        1. 分析新闻整体内容情绪
        2. 匹配情绪词典关键词，对情绪进行深入理解与分析
        3. 输出情绪分析跟对应的五档制评级结果（非常消极/比较消极/中性/比较积极/非常积极）

        **情绪评级逻辑**
        1. 语义匹配：忽略情绪词典中与新闻不相关的情绪词，保留有效情绪词
        2. 评级调整：新闻整体语义优先，情绪词典辅助修正

        **示例说明**
        - 示例1：
        新闻文本：实现净利润同比增长137.98%，单季度的盈利规模超过中信证券成为业内第一
        情绪词典：profitability:0.6,profit:0.8
        情绪分析:①语义信息为净利润同比大幅增长137.98%及单季度盈利规模跃居行业第一，均体现超预期的盈利能力突破；②关键情绪词调整：“净利润”（匹配profit）和“盈利”（匹配profitability）共同强化积极方向。两重强信号叠加符合最高档“非常积极”。
        情绪评级:非常积极

        - 示例2：
        新闻文本：中国的A股定位反而是比较便宜的，外资从全球定价认为我们非常有吸引力
        情绪词典：mispricing:-0.4,advantage:0.7
        情绪分析:①语义信息为A股估值被强调为“便宜”及外资认可其全球定价吸引力，隐含市场价值被低估的积极信号；②关键情绪词调整：未直接匹配词典中的“mispricing”或“advantage”，但“便宜”隐含定价偏离逻辑（映射mispricing方向），“有吸引力”间接呼应优势（advantage方向）。由于缺乏词典强匹配项限制进一步上调空间，整体乐观基调符合“比较积极”。
        情绪评级:比较积极

        - 示例3：
        新闻文本：美国信奉自由市场经济理念，主张靠无形的手调整经济活动
        情绪词典：free:0.2,immateriality:-0.2
        情绪分析:①语义信息为对美国经济理念的中性陈述，既未直接关联中国市场优劣，也未体现政策对华影响；②关键情绪词调整：“自由”（匹配free:+0.2）与“无形”（匹配immateriality:-0.2）存在方向冲突，但文本未实际使用“immateriality”原词（仅隐含“无形的手”概念），语义匹配强度不足。陈述性内容缺乏明确情绪导向，只是陈述事件，符合中性基准。
        情绪评级:中性

        - 示例4：
        新闻文本：我们投入的前期费用谁来承担
        情绪词典：invest:0.3
        情绪分析:①语义信息为对前期费用承担主体的质疑，隐含投入成本未被消化的潜在风险，传递财务负担不确定性的负面情绪；②关键情绪词调整：“投入”（匹配invest:+0.3）存在方向性冲突，因文本中“投入”实际指向成本分摊压力而非正向投资预期，情绪词得分被整体语义逆向修正。中性词主导+隐含担忧的复合信号符合低度负面评分档位“比较消极”。
        情绪评级:比较消极

        - 示例5：
        新闻文本：饱受美国次贷危机冲击的华尔街再次风云突变
        情绪词典：crash:-0.9,meltdown:-0.8
        情绪分析:①语义信息为华尔街受次贷危机冲击引发的市场动荡，此类全球金融中心的不稳定通常导致跨国资本避险情绪上升，对中国市场构成外溢风险；②关键情绪词调整：未直接匹配“crash”或“meltdown”，但“次贷危机冲击”与“风云突变”共同映射系统性风险（贴近meltdown的-0.8方向），叠加事件严重性突破常规调整范畴。极端负面事件的整体语义强度主导评分，因此是“非常消极”。
        情绪评级:非常消极


        **其他说明**
        - 情绪词典的分值仅作辅助作用
        - 情绪评级必须是五档制选择，不得出现非常消极/比较消极/中性/比较积极/非常积极之外的情绪等级
        - 输出格式：{"情绪分析":"...","情绪评级":"..."}
        }
        现在，请你开始分析并按照要求输出结果：
        新闻文本：{{新闻文本}}
        情绪词典：{{情绪词}}"""
        
        final_df['qwen_input'] = final_df.apply(
            lambda row: prompt2.replace('{{新闻文本}}', row['cut_news'] + ('。' if not row['cut_news'].endswith('。') else ''))
                    .replace('{{情绪词}}', row['sentiment_dict_v2']),
            axis=1
        )
        final_df.to_excel('/root/onethingai-fs/mydata/process_data/Macro_report_70393_rag_processed.xlsx')


############# ==========================  添加的额外处理 ==========================  #############


        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise
    

def load_sentiment_words(sentiment_dict_path):
    """
    加载情感词典并构造prompt
    
    参数:
    sentiment_dict_path: 情感词典路径
    
    返回:
    list: 改写后的prompt列表
    list: 原始单词列表
    list: 情感分数列表
    """
    prompts = []
    original_words = []
    scores = []
    
    with open(sentiment_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, score = line.strip().split('\t')
            prompt = f"与新闻内容最相关的单词是“{word}”"
            prompts.append(prompt)
            original_words.append(word)
            scores.append(float(score))
    
    return prompts, original_words, scores

def build_faiss_index(embeddings):
    """
    构建FAISS索引，优先使用GPU
    """
    start_time = time.time()
    print(f"开始构建索引，向量维度: {embeddings.shape}")
    
    # 确保数据类型正确
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    print(f"向量数据类型: {embeddings.dtype}")
    
    # 添加GPU内存监控
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
        print(f"清理GPU缓存后内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    dimension = embeddings.shape[1]
    
    # 创建CPU索引
    index = faiss.IndexFlatIP(dimension)
    
    # 如果有GPU，转换为GPU索引
    if faiss.get_num_gpus() > 0:
        print(f"gpu个数{faiss.get_num_gpus()}，使用GPU构建FAISS索引")
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("成功创建GPU索引")
            
            index.add(embeddings)
            print("成功添加向量到索引")
            
            end_time = time.time()
            print(f"索引构建完成，耗时: {end_time - start_time:.2f}秒")
            return index
        except Exception as e:
            print(f"GPU索引构建失败: {str(e)}")
            print("回退到CPU索引")
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            end_time = time.time()
            print(f"CPU索引构建完成，耗时: {end_time - start_time:.2f}秒")
            return index
    else:
        print("使用CPU构建FAISS索引")
        index.add(embeddings)
        end_time = time.time()
        print(f"索引构建完成，耗时: {end_time - start_time:.2f}秒")
        return index

# 添加内存管理函数
def clear_gpu_memory():
    """
    清理GPU内存
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def monitor_memory():
    """
    监控内存使用情况
    """
    process = psutil.Process()
    print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    # 模型路径
    model_path = '/root/onethingai-fs/mymodels/BAAI/bge-m3'  # 更改为BGE-M3模型
    reranker_path = '/root/onethingai-fs/mymodels/BAAI/bge-reranker-v2-m3'
    
    # 文件路径
    news_file_path = '/root/onethingai-fs/mydata/newspaper/Macro_report_70393_mid.xlsx'
    lda_key_words_path = '/root/LLM_news_emo_analyze/DATA/lda_key_words_0509.xlsx'
    sentiment_dict_path = '/root/LLM_news_emo_analyze/DATA/process_senti_dic_EN2.txt'
    output_file_path = '/root/onethingai-fs/mydata/newspaper/Macro_report_70393_rag_unprocessed.xlsx'
    
    # 调用主处理函数
    process_news_data(news_file_path, lda_key_words_path, sentiment_dict_path, 
                     model_path, reranker_path, output_file_path)