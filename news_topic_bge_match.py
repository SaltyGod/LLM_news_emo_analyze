from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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
    embedding_1 (torch.Tensor): 第一个嵌入表示。
    embeddings (torch.Tensor): 其他嵌入表示的张量。

    返回:
    torch.Tensor: 相似度列表。
    """
    cosine_similarities = F.cosine_similarity(embedding_1.unsqueeze(0), embeddings)
    return cosine_similarities

if __name__ == "__main__":
    prompt_1 = "这是第一个自定义的提示"
    other_prompts = [
        "这是第二个自定义的提示",
        "这是第三个自定义的提示",
        "这是第四个自定义的提示",
        "这是第五个自定义的提示"
    ]
    model_path = '/root/.cache/LLMS/hub/BAAI/bge-large-zh-v1___5'  # BGE-large-zh-v1.5模型
    tokenizer, model = load_model(model_path)

    # 获取所有提示的嵌入
    all_prompts = [prompt_1] + other_prompts
    embeddings = get_embeddings(all_prompts, tokenizer, model)

    # 计算prompt_1与其他提示的相似度
    similarity_scores = calculate_cosine_similarity(embeddings[0], embeddings[1:])

    # 找出相似度最高的提示及其相似度值
    max_similarity_index = torch.argmax(similarity_scores).item()
    max_similarity_score = similarity_scores[max_similarity_index].item()
    most_similar_prompt = other_prompts[max_similarity_index]

    print("与prompt_1最相似的提示是:", most_similar_prompt)
    print("相似度值:", max_similarity_score)