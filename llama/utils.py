import os
import re
import random

def load_template(type):
    with open(os.path.join('prompt_templates', type + '.md'), 'r') as f:
        return f.read()


def check_response(response):
    punctuation_marks = {'.', '!', '?', ',', ';', ':', '-', '...', '"', "'", ')', ']', '}', '>', '`'}
    response = response.rstrip()
    if not response or response[-1] not in punctuation_marks:
        print('punctuation_marks failed', response[-10:])
        return False
    stopwords = ['revised', 'revision', 'response', 'version', 'paragraph']
    part = response.split('\n')
    part = part[:2] + part[-3:]
    part = '\n'.join(part)
    for word in stopwords:
        if word in part:
            print('contains stopwords failed', word)
            return False
    response = response.split()
    n_gram = {}
    count = 10
    for i in range(len(response)):
        if i >= count - 1:
            substr = response[i-count+1:i+1]
            x = n_gram.get(tuple(substr), 0)
            n_gram[tuple(substr)] = x + 1
            if x > 0:
                print('n_gram failed:', substr)
                return False
    return True


def drop_short(responses, key):
    lengths = []
    for data in responses:
        lengths.append(len(data[key]))
    lengths = sorted(lengths)
    select_index = []
    for ith, data in enumerate(responses):
        length = len(data[key])
        rank = 1 - lengths.index(length) / len(lengths)
        if random.random() > 2 * rank ** 3:
            select_index.append(ith)
    responses = responses.select(select_index)
    return responses


def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text)
    english_words = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text).split()
    total_word_count = len(chinese_characters) + len(english_words)
    return total_word_count

