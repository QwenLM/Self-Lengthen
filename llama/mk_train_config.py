import yaml
import argparse
import os
import config


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def write_yaml_file(file_path, data):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)


def mk_train_config(iter):
    os.makedirs('train_config', exist_ok=True)

    gen_sft = read_yaml_file('train_config_template/template_gen_sft.yaml')
    gen_sft['model_name_or_path'] = config.model_name_or_path
    if iter != 1:
        gen_sft['adapter_name_or_path'] = f'results/iter_{iter - 1}/lora/gen_sft'
    gen_sft['output_dir'] = f'results/iter_{iter}/lora/gen_sft'
    gen_sft['tokenized_path'] = f'results/iter_{iter}/tokenized_data/gen_sft'
    write_yaml_file('train_config/gen_sft.yaml', gen_sft)

    ext_sft = read_yaml_file('train_config_template/template_ext_sft.yaml')
    ext_sft['model_name_or_path'] = config.model_name_or_path
    if iter != 1:
        ext_sft['adapter_name_or_path'] = f'results/iter_{iter - 1}/lora/ext_sft'
    ext_sft['dataset'] = 'ext_sft'
    ext_sft['output_dir'] = f'results/iter_{iter}/lora/ext_sft'
    ext_sft['tokenized_path'] = f'results/iter_{iter}/tokenized_data/ext_sft'
    write_yaml_file('train_config/ext_sft.yaml', ext_sft)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=1)
    args = parser.parse_args()
    mk_train_config(args.iter)
