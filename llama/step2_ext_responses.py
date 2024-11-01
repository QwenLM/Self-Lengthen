from utils import load_template, check_response, drop_short, count_words
from llm_client import get_response, get_completion
from tqdm import tqdm
from datasets import Dataset
import concurrent.futures
import argparse
import config
import os


def extend_response(prompt, response, ext_model):
    inputs = response
    best_output = response
    extend_template = load_template('extend_response')
    prompt_prefix = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
    prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    for _ in range(3):
        lines = inputs.split('\n\n')
        if len(lines) <= 2:
            lines = inputs.split('\n')
        former = '\n\n'.join(lines[:len(lines) // 2])
        former_input = extend_template.format(prompt=prompt, response=former)
        former_output = get_response(former_input, ext_model)

        former_output = former_output.split('\n\n')
        former_output = former_output[:len(former_output) * 2 // 3]
        former_output = '\n\n'.join(former_output)

        latter_input = prompt_prefix + extend_template.format(prompt=prompt, response=inputs) + prompt_suffix + former_output
        latter_output = get_completion(latter_input, ext_model)
        output = former_output + latter_output

        # print(f'-----\n\n{prompt}\n---\n{response}\n---\n{output}\n---\n\n-----\n\n')
        print(f'length: {count_words(inputs)} -> {count_words(output)}')
        if check_response(output) and count_words(output) > count_words(best_output):
            inputs = output
            best_output = output
    print(f'final length: {count_words(response)} -> {count_words(best_output)}')
    if count_words(best_output) > 1.2 * count_words(response):
        print('Extended successfully.')
        return best_output
    return None


def gen_extended_responses(initial_responses, ext_model):
    extended_responses = {
        'prompt': [],
        'initial_response': [],
        'extended_response': [],
    }

    def length_extend_task(prompt, initial_response, retry_times=20):
        if retry_times == 0:
            return None
        extended_response = extend_response(prompt, initial_response, ext_model)
        if extended_response is not None:
            return prompt, initial_response, extended_response
        return length_extend_task(prompt, initial_response, retry_times - 1)

    print('Begin extending responses...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(length_extend_task, **initial_responses) for initial_responses in initial_responses for _ in range(2)}
        count = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result is not None:
                    extended_responses['prompt'].append(result[0])
                    extended_responses['initial_response'].append(result[1])
                    extended_responses['extended_response'].append(result[2])
                count += 1
                if count / len(futures) > 0.95:
                    extended_responses = Dataset.from_dict(extended_responses)
                    extended_avg_length = sum(count_words(response) for response in extended_responses['extended_response']) / len(extended_responses)
                    print(f'Before dropping length: {extended_avg_length}')
                    print(extended_responses)
                    extended_responses = drop_short(extended_responses, 'extended_response')
                    initial_avg_length = sum(count_words(response) for response in extended_responses['initial_response']) / len(extended_responses)
                    extended_avg_length = sum(count_words(response) for response in extended_responses['extended_response']) / len(extended_responses)
                    print('Extended responses written.')
                    print(f'Average length: {initial_avg_length} -> {extended_avg_length}')
                    print(extended_responses)
                    extended_responses.save_to_disk(f'results/iter_{args.iter}/extended_responses')
                    os.system("pkill -9 -e -f step2_ext_responses.py")
            except Exception as exc:
                print(f"generated an exception: {exc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    initial_responses = Dataset.load_from_disk(f'results/iter_{args.iter}/initial_responses')
    if args.iter == 1:
        ext_model = config.model_name_or_path
    else:
        ext_model = 'ext_sft'
    initial_responses = initial_responses.shuffle(seed=42)
    gen_extended_responses(initial_responses, ext_model)
