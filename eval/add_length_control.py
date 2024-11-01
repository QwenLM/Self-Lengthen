import json
import re
import random
import math
import torch
from langdetect import detect

from cn2an import an2cn

random.seed(1234)

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text)
    english_words = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text).split()
    total_word_count = len(chinese_characters) + len(english_words)
    return total_word_count

def num_round(x, num=1000):
    if x <= 10:
        return x
    if x < 100:
        num = 10
    if x == 100:
        return 100, 90, 100
    if x < 1000:
        num = 100
    if x == 1000:
        return 1000, 900, 1000
    return round(x/num) * num, math.floor(x/num) * num, math.ceil(x/num) * num

template = {
    "zh": """请帮助我在用户提供的原始查询中加入字数限制，要求为“{}”，但不要修改查询的其他部分。具体要求如下：

- 如果用户的原始查询中已经包含了类似限制，也将其修改为上述要求。
- 当添加限制时，注意保证其形式、表达和放置位置的多样化，以避免显得生硬或重复。
- 直接返回最终结果，无需额外的格式和内容。""",
    "en": """Please assist in integrating a word limit into the user's original query with the specified requirement of "{}". It is important to keep the rest of the query unchanged. Please follow these guidelines:

- If the original query from the user already explicitly includes a word limit, adjust it to meet the aforementioned requirements.
- When adding a word limit, ensure the phrasing, expression, and placement of the limit are varied to maintain natural and diverse language.
- Directly return the final result with no additional formatting or content."""
}

constraint = {
    "zh": ['大约{}字', '{}字至{}字', '小于{}字', '大于{}字'],
    "en": ['around {} words', '{}-{} words', 'more than {} words', 'less than {} words'],
}

data = []
num = 0
filename = 'long'

data = []
lengths = []
for d in open(filename + '.jsonl'):
    d = json.loads(d.strip())
    length = count_words(d['session'][0]['response'])
    if length < 2000 or length >= 8000:
        continue
    lang = 'zh' if 'zh' in detect(d['session'][0]['response']) else 'en'
    num = 2000 if random.random() < 0.2 else 1000
    round_length, min_length, max_length = num_round(length, num)
    max_length = min(max_length, 8000)
    min_length = max(min_length, 2000)
    if random.random() < 0.5 and lang == 'zh':
        round_length, min_length, max_length = an2cn(round_length), an2cn(min_length), an2cn(max_length)

    r = random.random()
    if r < 0.4:
        target_length = constraint[lang][0].format(round_length)
    elif 0.4 <= r and r < 0.65:
        target_length = constraint[lang][1].format(min_length, max_length)
    elif 0.65 <= r and r < 0.75:
        target_length = constraint[lang][2].format(min_length)
    else:
        target_length = constraint[lang][3].format(max_length)
    lengths.append(length)
    data.append({"eval_args": {"system_str": template[lang].format(target_length)}, "length": length, "prompt": d['session'][0]['query'], "output": d['session'][0]['response'], "tags": "core,generation", "task": f"generation"})

print(torch.histogram(torch.tensor(lengths).float(), bins=6, range=(2000,8000)))

filename = filename + '_tobeaddlc.jsonl'
with open(filename, 'w', encoding='utf-8') as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')