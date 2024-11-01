from utils import check_response, drop_short, count_words
from llm_client import get_response
from tqdm import tqdm
import concurrent.futures
from datasets import Dataset
import argparse
import config
import os


def gen_initial_responses(instructions, gen_model):
    initial_responses = {
        'prompt': [],
        'initial_response': [],
    }

    def gen_initial_response_task(prompt, retry_times=20):
        if retry_times == 0:
            return None
        initial_response = get_response(prompt, gen_model)
        initial_response = initial_response.strip()
        if not check_response(initial_response):
            return gen_initial_response_task(prompt, retry_times - 1)
        print(f'length: {count_words(initial_response)}')
        return prompt, initial_response

    print('Begin writing initial responses...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(gen_initial_response_task, instruction['prompt']) for instruction in instructions for _ in range(2)}
        count = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result is not None:
                    initial_responses['prompt'].append(result[0])
                    initial_responses['initial_response'].append(result[1])
                    if len(initial_responses['initial_response']) % 100 == 0:
                        dataset = Dataset.from_dict(initial_responses)
                        dataset.save_to_disk(f'results/iter_{args.iter}/initial_responses')
                count += 1
                if count / len(futures) > 0.95:
                    initial_responses = Dataset.from_dict(initial_responses)
                    average_length = sum(count_words(response) for response in initial_responses['initial_response']) / len(initial_responses)
                    print(f'Before dropping length: {average_length}')
                    print(initial_responses)
                    initial_responses = drop_short(initial_responses, 'initial_response')
                    average_length = sum(count_words(response) for response in initial_responses['initial_response']) / len(initial_responses)
                    print(f'Initial responses generated. average length: {average_length}')
                    print(initial_responses)
                    initial_responses.save_to_disk(f'results/iter_{args.iter}/initial_responses')
                    os.system("pkill -9 -e -f step1_gen_initial_responses.py")
            except Exception as exc:
                print(f"generated an exception: {exc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    instructions = Dataset.load_from_disk(f'results/iter_{args.iter}/instructions')
    if args.iter == 1:
        gen_model = config.model_name_or_path
    else:
        gen_model = f'gen_sft'
    print(instructions)
    gen_initial_responses(instructions, gen_model)
