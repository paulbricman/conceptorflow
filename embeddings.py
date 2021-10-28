from transformers import AutoTokenizer, AutoModel
import torch
import pickle


def save_token_embeddings(input, filename):
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    data = {}
    for token_id_idx, token_id in enumerate(encoded_input['input_ids'][0].numpy()):
        if token_id not in data.keys():
            data[token_id] = [model_output[0][0][token_id_idx].numpy()]
        else:
            data[token_id] += [model_output[0][0][token_id_idx].numpy()]

    data.pop(101, None)
    data.pop(102, None)
    
    pickle.dump(data, open(filename, 'wb+'))