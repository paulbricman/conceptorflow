from datasets import load_dataset
import re
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pickle
import os
from conceptors import Conceptor


def generate_conceptors(state_clouds_directory, conceptor_directory):
    for filename in os.listdir(state_clouds_directory):
        print('(*) working on:', filename)
        state_cloud = pickle.load(open(state_clouds_directory + '/' + filename, 'rb'))
        c = Conceptor().from_states(state_cloud)
        pickle.dump(c.conceptor_matrix, open(conceptor_directory + '/' + filename, 'wb+'))


def generate_state_clouds(keywords):
    print('(*) Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    print('\n\n(*) Loading dataset...')
    dataset = load_dataset('bookcorpusopen')

    for keyword in keywords:
        print('\n\n(*) Extracting state cloud of:', keyword)
        sentences = get_sentences(keyword, dataset)
        embeddings = get_embeddings(keyword, sentences, tokenizer, model)
        pickle.dump(embeddings, open('./clouds/' + keyword + '.pickle', 'wb+'))


def get_sentences(keyword, dataset, context_size=300):

    pattern = re.compile(keyword)
    raw_contexts = []

    for doc in dataset['train']['text']:
        for match in pattern.finditer(doc):
            start = max(0, match.start() - context_size)
            end = min(len(doc), match.end() + context_size)
            raw_context = doc[start:end].replace('\n', ' ')
            raw_contexts += [raw_context]

    print('(*)', len(raw_contexts), 'raw contexts identified')

    sentences = []

    for raw_context in raw_contexts:
        sents = sent_tokenize(raw_context)
        if len(sents) > 2:
            sents = sents[1:-1]
            sents = [e.lower() for e in sents if keyword in e]
            sentences += sents
        
    sentences = list(set(sentences))
    print('(*)', len(sentences), 'unique sentences containing keyword')

    return sentences


def get_embeddings(keyword, sentences, tokenizer, model):
    encoded_keyword = tokenizer(keyword, padding=True, truncation=True, return_tensors='pt')['input_ids'][0][1:-1]
    embeddings = []

    for e in sentences:
        encoded_sentence = tokenizer(e, padding=True, truncation=True, return_tensors='pt')
        if is_sublist(encoded_keyword, encoded_sentence['input_ids'][0]):
            start_keyword = get_sublist_idx(encoded_keyword, encoded_sentence['input_ids'][0])
            if start_keyword is not None:
                end_keyword = start_keyword + len(encoded_keyword)
                with torch.no_grad():
                    model_output = model(encoded_sentence['input_ids'])
                    embeddings += [np.mean(model_output[0][0][start_keyword:end_keyword].numpy(), axis=0)]

    print('(*)', len(embeddings), 'embeddings extracted')
        
    return embeddings
    

def is_sublist(ls1, ls2):
    def get_all_in(one, another):
        for element in one:
            if element in another:
                yield element

    for x1, x2 in zip(get_all_in(ls1, ls2), get_all_in(ls2, ls1)):
        if x1 != x2:
            return False

    return True


def get_sublist_idx(x, y):
    l1, l2 = len(y), len(x)
    for i in range(l1):
        if torch.equal(y[i:i+l2], x):
            return i