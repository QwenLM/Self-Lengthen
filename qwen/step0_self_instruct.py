from utils import load_template, count_words
from llm_client import get_completion, get_response
import random
from tqdm import tqdm
import concurrent.futures
import threading
from datasets import Dataset, concatenate_datasets, load_from_disk
import argparse


def self_instruct(count=8):
    instructions = load_from_disk('prompt_templates/prompts_train')['prompt']
    length = len(instructions)

    template = load_template('self_instruct')
    judge_template = load_template('judge_prompt')
    lock = threading.Lock()

    def self_instruct_task(retry_times=10):
        if retry_times == 0:
            print('Failed to generate a valid instruction.')
            return None
        nonlocal instructions, template, lock
        with lock:
            examples = random.sample(instructions, 2)
        input = template.format(prompt1=examples[0], prompt2=examples[1])
        output = get_completion(input).strip()
        is_suitable = get_response(judge_template.format(prompt=output)).strip()
        if is_suitable == '是' and 3 < count_words(output) < 500:
            with lock:
                instructions.append(output)
            return output
        return self_instruct_task(retry_times - 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(self_instruct_task) for _ in range(count)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
            except Exception as exc:
                print(f"generated an exception: {exc}")

    dataset = {
        'prompt': instructions[length:],
    }
    dataset = Dataset.from_dict(dataset)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    instructions_list = []
    print('Begin self instructing...')
    for i in range(args.count // 500):
        instructions = self_instruct(count=500)
        instructions_list.append(instructions)
    instructions = self_instruct(count=args.count % 500)
    instructions_list.append(instructions)
    instructions = concatenate_datasets(instructions_list)
    print(instructions)
    print(instructions['prompt'][:10])
    instructions.save_to_disk(f'results/iter_{args.iter}/instructions')
