from datasets import load_from_disk
from utils import load_template, count_words
import json
import argparse
import os
import random


def mk_gen_sft(extended_responses):
    gen_sft = []
    length = 0
    for data in extended_responses:
        response = data['extended_response']
        gen_sft.append({
            'instruction': data['prompt'],
            'input': '',
            'output': response
        })
        length += count_words(response)
    print(f'Finish making gen_sft data, {len(gen_sft)} samples in total. Average length: {length / len(gen_sft)}')
    with open('data/gen_sft.json', 'w') as f:
        json.dump(gen_sft, f)
    return gen_sft


def mk_ext_sft(extended_responses):
    extend_template = load_template('extend_response')
    length = 0
    ext_sft = []
    for data in extended_responses:
        initial_response = data['initial_response']
        extended_response = data['extended_response']
        initial_response = initial_response.split('\n')
        initial_response = [line for line in initial_response if random.random() < 0.8]
        initial_response = '\n\n'.join(initial_response)
        ext_sft.append({
            'instruction': extend_template.format(prompt=data['prompt'], response=initial_response),
            'input': '',
            'output': extended_response
        })
        length += count_words(extended_response)
    print(f'Finish making ext_sft data, {len(ext_sft)} samples in total. Average length: {length / len(ext_sft)}')
    with open('data/ext_sft.json', 'w') as f:
        json.dump(ext_sft, f)
    return ext_sft


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=1)

    args = parser.parse_args()
    extended_responses = load_from_disk(f'results/iter_{args.iter}/extended_responses')
    extended_responses = extended_responses.shuffle()
    print(extended_responses)

    os.makedirs(f'results/iter_{args.iter}/data', exist_ok=True)
    gen_sft = mk_gen_sft(extended_responses)
    with open(f'results/iter_{args.iter}/data/gen_sft.json', 'w') as f:
        json.dump(gen_sft, f)
    
    ext_sft = mk_ext_sft(extended_responses)
    with open(f'results/iter_{args.iter}/data/ext_sft.json', 'w') as f:
        json.dump(ext_sft, f)
