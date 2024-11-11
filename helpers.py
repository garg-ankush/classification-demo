# Function to get BERT embeddings
import torch

def get_bert_embeddings(texts, tokenizer, model, max_length=512):
    # Tokenize texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embeddings