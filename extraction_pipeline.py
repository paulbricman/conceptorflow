from datasets import load_dataset
import re
from nltk.tokenize import sent_tokenize


def get_contexts(keywords, context_size=200):
    dataset = load_dataset('bookcorpusopen')

    for keyword in keywords:
        pattern = re.compile(keyword)
        raw_contexts = []

        for doc in dataset['train'][:10]['text']:
            for match in pattern.finditer(doc):
                start = max(0, match.start() - context_size)
                end = min(len(doc), match.end() + context_size)
                raw_context = doc[start:end].replace('\n', ' ').lower()
                raw_contexts += [raw_context]

        print('(*)', len(raw_contexts), 'raw contexts identified')

        trimmed_contexts = []

        for raw_context in raw_contexts:
            sents = sent_tokenize(raw_context)
            if len(sents) == 1:
                trimmed_context = raw_context
            elif keyword in sents[0]:
                trimmed_context = ' '.join(sents[:-1])
            elif keyword in sents[-1]:
                trimmed_context = ' '.join(sents[1:])
            elif keyword in ' '.join(sents[1:-1]):
                trimmed_context = ' '.join(sents[1:-1])
            
            trimmed_contexts += [trimmed_context]
        
        return trimmed_contexts