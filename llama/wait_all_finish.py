import argparse
import os
import time


def wait_all_finish(iter):
    while True:
        if os.path.exists(f'results/iter_{iter}/extended_responses'):
            break
        time.sleep(10)

def wait_tokenizing(iter, config_file):
    config_file = config_file.split('/')[-1].split('.')[0]
    while True:
        time.sleep(10)
        if os.path.exists(f'results/iter_{iter}/tokenized_data/{config_file}'):
            break
    time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--phase', type=str, default='makingdata')
    parser.add_argument('--config_file', type=str, default='')
    args = parser.parse_args()
    print(f'waiting {args.phase} iter {args.iter}...')
    if args.phase == 'makingdata':
        wait_all_finish(args.iter)
    else:
        wait_tokenizing(args.iter, args.config_file)
