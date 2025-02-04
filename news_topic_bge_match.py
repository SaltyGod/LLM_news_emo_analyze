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

def calculate_cosine_similarity(embeddings):
    """
    计算两个嵌入的余弦相似度。

    参数:
    embeddings (torch.Tensor): 嵌入表示的张量。
    
    返回:
    float: 两个嵌入的余弦相似度。
    """
    cosine_similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return cosine_similarity.item()

if __name__ == "__main__":
    prompt_1 = "这是第一个自定义的提示"
    prompt_2 = "这是第二个自定义的提示"
    model_path = '/root/.cache/LLMS/hub/BAAI/bge-large-zh-v1___5' # BG-large-zh-v1.5模型
    
    tokenizer, model = load_model(model_path)
    embeddings = get_embeddings([prompt_1, prompt_2], tokenizer, model)
    
    similarity = calculate_cosine_similarity(embeddings)
    print("余弦相似度:", similarity)