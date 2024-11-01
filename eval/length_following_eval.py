import re
import argparse
import json

def count_words(text: str) -> int:
    chinese_characters = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text)
    english_words = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text).split()
    total_word_count = len(chinese_characters) + len(english_words)
    return total_word_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()
    with open(args.data_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    scores = {
        "about": [],
        "range": [],
        "above": [],
        "below": []
    }
    for data in dataset:
        target_length_str = data["constraint"]
        if data["type"] == "about":
            target_length = int(re.search(r"(\d+)", target_length_str).group(1))
            target_length_a = target_length * 0.8
            target_length_b = target_length * 1.2
        elif data["type"] == "range":
            match = re.search(r"(\d+)\D+(\d+)", target_length_str)
            target_length_a = int(match.group(1))
            target_length_b = int(match.group(2))
        elif data["type"] == "above":
            target_length_a = int(re.search(r"(\d+)", target_length_str).group(1))
            target_length_b = target_length_a * 1.5
        elif data["type"] == "below":
            target_length_b = int(re.search(r"(\d+)", target_length_str).group(1))
            target_length_a = target_length_b * 0.5
        print(target_length_str, target_length_a, target_length_b, data["type"])
        
        gen_length = count_words(data["response"])
        if gen_length >= target_length_a and gen_length <= target_length_b:
            score = 1
        elif gen_length < target_length_a:
            score = max(gen_length / target_length_a * 2 - 1, 0)
        elif gen_length > target_length_b:
            score = max(3 - gen_length / target_length_b * 2, 0)

        scores[data["type"]].append(score)

    for key in scores:
        print(f"{key}: {sum(scores[key]) / len(scores[key])}")
    print(f"Overall: {sum(sum(scores[key]) for key in scores) / sum(len(scores[key]) for key in scores)}")
