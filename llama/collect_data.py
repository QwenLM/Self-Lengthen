from datasets import load_from_disk, concatenate_datasets
import os
import json

response_list = []

for dirs in os.listdir('results'):
    try:
        initial_responses = load_from_disk(f'results/{dirs}/initial_responses')
        initial_responses = initial_responses.rename_column("initial_response", "response")
        response_list.append(initial_responses)
        extended_responses = load_from_disk(f'results/{dirs}/extended_responses')
        extended_responses = extended_responses.rename_column("extended_response", "response")
        response_list.append(extended_responses)
    except:
        pass

responses = concatenate_datasets(response_list)
print(responses)

json_dataset = []

for data in responses:
    json_dataset.append({'query': data['prompt'], 'response': data['response']})

with open('long.jsonl', 'w') as f:
    for data in json_dataset:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
