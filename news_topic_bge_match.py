from transformers import AutoTokenizer, AutoModel
import torch
import re
import torch.nn.functional as F
import pandas as pd

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
                prompt = f"新闻的主题是{theme_category}，关键词是{'、'.join(keywords)}"
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

# 处理新闻文本并计算相似度
def process_news_data(news_file_path, lda_key_words_path, model_path, output_file_path, similarity_threshold=0.37):
    """
    处理新闻数据，进行主题匹配。
    
    参数:
    news_file_path (str): 新闻文本文件路径
    lda_key_words_path (str): 主题关键词Excel文件路径
    model_path (str): 模型路径
    output_file_path (str): 输出文件路径
    similarity_threshold (float): 相似度阈值，默认为0.2
    """
    try:
        # 加载模型和分词器
        tokenizer, model = load_model(model_path)
        
        # 读取新闻文本的Excel文件
        df = pd.read_excel(news_file_path, nrows=10)
        if df.empty:
            raise ValueError("新闻文件为空")
        
        # 获取主题提示
        key_topic_list = get_theme_prompts(lda_key_words_path)
        print(f"成功加载 {len(key_topic_list)} 个主题提示")
        
        # 获取主题提示的嵌入
        key_topic_embeddings = get_embeddings(key_topic_list, tokenizer, model)
        print(f"主题嵌入向量形状: {key_topic_embeddings.shape}")
        
        # 遍历每一行
        for index, row in df.iterrows():
            try:
                if pd.isna(row['NewsContent']):
                    print(f"跳过第 {index} 行：新闻内容为空")
                    continue
                    
                news_content = str(row['NewsContent'])
                sentences = split_sentences(news_content)
                
                if not sentences:
                    print(f"跳过第 {index} 行：分句结果为空")
                    continue
                
                # 存储分隔后的句子
                sentence_dict = {f"news_prompt{i+1}": f"\"{sentence}\"" 
                               for i, sentence in enumerate(sentences)}
                df.at[index, 'NewsContent_cut'] = str(sentence_dict)
                
                # 存储每个句子的主题匹配结果
                topic_match_dict = {}
                for sentence in sentences:
                    # 获取句子的嵌入
                    sentence_embedding = get_embeddings([sentence], tokenizer, model)
                    
                    # 计算相似度
                    similarity_scores = calculate_cosine_similarity(sentence_embedding, 
                                                                 key_topic_embeddings)
                    
                    # 找出相似度最高的主题
                    max_similarity_index = torch.argmax(similarity_scores).item()
                    similarity_score = similarity_scores[max_similarity_index].item()
                    
                    # 只有当相似度大于阈值时才添加到结果中
                    if similarity_score > similarity_threshold:
                        most_similar_topic = key_topic_list[max_similarity_index]
                        theme_category = extract_theme_category(most_similar_topic)
                        
                        topic_match_dict[f"\"{sentence}\""] = {
                            "topic": theme_category,
                            "similarity": similarity_score
                        }
                        print(topic_match_dict)
                
                # 存储到新的列
                df.at[index, 'cut_news_topic_match'] = str(topic_match_dict)
                print(f"成功处理第 {index} 行")
                
            except Exception as e:
                print(f"处理第 {index} 行时出错: {str(e)}")
                continue
        
        # 保存结果到新的Excel文件
        df.to_excel(output_file_path, index=False)
        print(f"处理完成，结果已保存到 {output_file_path}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 模型路径
    model_path = '/root/.cache/LLMS/hub/BAAI/bge-large-zh-v1___5'
    
    
    # 新闻文本文件路径
    news_file_path = './DATA/news_test_data.xlsx'
    
    # 主题提示文件路径
    lda_key_words_path = './DATA/lda_key_words.xlsx'
    
    # 输出文件路径
    output_file_path = './output_result/DATA/news_data_with_topics.xlsx'
    
    # 调用主处理函数
    process_news_data(news_file_path, lda_key_words_path, model_path, output_file_path)
