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

def load_model(model_path):
    """
    加载模型和分词器。

    参数:
    model_path (str): 模型的路径。

    返回:
    tokenizer: 加载的分词器。
    model: 加载的模型。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
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
    # Tokenize custom prompts
    encoded_input = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

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
    sentences = re.split(r'[！。？；]', text)
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
    加载reranker模型
    """
    tokenizer = MSTokenizer.from_pretrained(reranker_path)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_path)
    model.eval()
    return tokenizer, model

def compute_reranker_score(model, tokenizer, text1, text2):
    """
    使用reranker计算两段文本的相关性得分
    """
    with torch.no_grad():
        inputs = tokenizer([text1], [text2], padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs).logits.view(-1,).float()
    return scores.item()

def hybrid_sentiment_search(sentence, faiss_index, sentiment_prompts, original_words, sentiment_scores,
                          reranker_model, reranker_tokenizer, bge_tokenizer, bge_model, 
                          first_stage_threshold=0.4, top_k=10):
    """
    两阶段搜索：FAISS快速筛选 + Reranker精确排序
    """
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
                                      return_tensors='pt', max_length=512)
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

@lru_cache(maxsize=1024)
def get_cached_embeddings(text, tokenizer, model, is_batch=False):
    """
    带缓存的向量计算
    """
    if is_batch:
        return get_embeddings(text, tokenizer, model)
    return get_embeddings([text], tokenizer, model)

def batch_get_embeddings(texts, tokenizer, model, batch_size=32):
    """
    批量计算向量
    """
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = get_embeddings(batch_texts, tokenizer, model)
        embeddings_list.append(batch_embeddings)
    return torch.cat(embeddings_list, dim=0)

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
        "不构成任何投资建议", "亦不代表平台观点", "请投资人独立判断和决策",
        "中报", "图片来源"
    ]
    
    # 检查句子长度
    if len(sentence.strip()) < 5:
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

def process_news_data(news_file_path, lda_key_words_path, sentiment_dict_path, 
                     model_path, reranker_path, output_file_path, 
                     similarity_threshold=0.42, batch_size=32):
    try:
        # 加载BGE-large模型
        bge_tokenizer, bge_model = load_model(model_path)
        
        # 加载reranker模型
        reranker_tokenizer, reranker_model = load_reranker(reranker_path)
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
        df = pd.read_excel(news_file_path)
        df = df.sample(100,random_state=42)
        if df.empty:
            raise ValueError("新闻文件为空")
        
        # 遍历处理每一行
        for index, row in tqdm(df.iterrows(), desc="处理新闻"):
            try:
                if pd.isna(row['NewsContent']):
                    continue
                
                news_content = str(row['NewsContent'])
                sentences = split_sentences(news_content)
                
                if not sentences:
                    continue
                
                # 存储有效的句子和处理结果
                valid_sentences = {}
                topic_match_dict = {}
                
                # 添加句子处理进度条
                for i, sentence in enumerate(tqdm(sentences, desc=f"处理第 {index} 行的句子", leave=False)):
                    try:
                        result = process_single_sentence(
                            sentence, faiss_index, sentiment_prompts, original_words, 
                            sentiment_scores, reranker_model, reranker_tokenizer,
                            bge_tokenizer, bge_model, key_topic_embeddings, 
                            key_topic_list, similarity_threshold
                        )
                        if result:
                            valid_sentences[f"news_prompt{i+1}"] = sentence
                            topic_match_dict[sentence] = result
                    except Exception as e:
                        print(f"处理句子失败 (3次重试后): {str(e)}")
                        continue
                
                # 只有当有有效结果时才更新DataFrame
                if valid_sentences and topic_match_dict:
                    df.at[index, 'NewsContent_cut'] = json.dumps(valid_sentences, ensure_ascii=False)
                    df.at[index, 'cut_news_topic_match'] = json.dumps(topic_match_dict, ensure_ascii=False)
                    print(f"第 {index} 行处理结果:", topic_match_dict)
                
            except Exception as e:
                print(f"处理第 {index} 行时出错: {str(e)}")
                continue
        
        # 保存结果
        df.to_excel(output_file_path, index=False)
        print(f"处理完成，结果已保存到 {output_file_path}")
        
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

if __name__ == "__main__":
    # 模型路径
    model_path = '/root/.cache/LLMS/hub/BAAI/bge-large-zh-v1___5'
    reranker_path = '/root/.cache/LLMS/hub/BAAI/bge-reranker-v2-m3'
    
    # 文件路径
    news_file_path = './DATA/news_test_data.xlsx'
    lda_key_words_path = './DATA/lda_key_words2.xlsx'
    sentiment_dict_path = './DATA/process_senti_dic_EN2.txt'
    output_file_path = './output_result/news_data_with_topics_0225.xlsx'
    
    # 调用主处理函数
    process_news_data(news_file_path, lda_key_words_path, sentiment_dict_path, 
                     model_path, reranker_path, output_file_path)