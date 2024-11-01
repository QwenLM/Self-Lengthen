from openai import OpenAI
import config


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_response(prompt, model=config.model_name_or_path):
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stop=["<|eot_id|>", "\n\n---", "\n\n\n", "\n---", "---", '<|im_end|>'],
        max_tokens=16384,
        top_p=0.8,
        temperature=0.7,
        extra_body={
            'repetition_penalty': 1.1,
            'top_k': 20,
        }
    )
    response = chat_response.choices[0].message.content
    return response


def get_completion(prompt, model=config.model_name_or_path):
    chat_response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=16384,
        stop=["<|eot_id|>", "\n\n---", "\n\n\n", "\n---", "---", '<|im_end|>', '\n指令4'],
        top_p=0.8,
        temperature=0.7,
        extra_body={
            'repetition_penalty': 1.1,
            'top_k': 20,
        }
    )
    response = chat_response.choices[0].text
    return response


if __name__ == '__main__':
    print(get_response('Hello, how are you?'))