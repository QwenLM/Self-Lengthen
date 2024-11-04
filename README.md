# Language Models Can Self-Lengthen to Generate Long Texts

*Shanghaoran Quan, Tianyi Tang, Bowen Yu, An Yang, Dayiheng Liu, Bofei Gao, Jianhong Tu, Yichang Zhang, Jingren Zhou, Junyang Lin*

Qwen Team, Alibaba Inc.

---

## :sparkles: Overview

This repository contains core implementations of the **Self-Lengthen**, proposed by [Language Models Can Self-Lengthen to Generate Long Texts](https://arxiv.org/abs/2410.23933).

**Self-Lengthen** is a novel and effective data-driven technique for extrapolating **long output**, designed to stimulate long-generation ability from scratch using only the LLM's intrinsic knowledge and skills. This is achieved by iteratively self-lengthening the output and inductively self-aligning to generate increasingly longer texts. By applying **Self-Lengthen**, we successfully increased the maximum output length of Qwen from 1,000 words to 8,000 words.

- **Low resource**: Self-Lengthen does not require high-quality human-written text; only a set of seed user long output instructions is needed.

- **Intrinsic ability**: Self-Lengthen utilizes only the seed LLM's intrinsic knowledge and skills, without any form of distillation from stronger LLMs.

- **Free form**: Self-Lengthen can generate suitable responses to a wide range of long output instructions and is not confined to strictly structured formats.

---

## :rocket: Data Synthesis of Self-Lengthen

### :wrench: Getting Started

1. Clone this repository.
1. We have made several crucial changes in `FastChat` project to support vLLM LoRA requests and extra decoding parameters like `repetition_penalty` and `top_k`. Please clone [this repository](https://github.com/quanshr/FastChat/tree/self-lengthen) and run `pip install -e ".[model_worker,webui]"` to install.
1. Run `pip install -r requirements.txt` to install other required packages.
1. The repository contains code for both Qwen and LLaMA models to apply Self-Lengthen. Both codebases share the same main structure with only slight differences (e.g., chat template, prompts, filter conditions) to fit their respective properties. Navigate to either the Qwen or LLaMA directory to proceed with the following steps.

### Running the Code

The code supports both multi-node and single-node execution. To run:

1. Change the `base_model` variable to your base model *path* at the top of `run.sh`, `run_server.sh`, `run_collect_data.sh`, and `config.py`.
1. Ensure that `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT` are set to the appropriate values on your cluster (`WORLD_SIZE` is 1 and `RANK` is 0 for a single node).
1. On each node, execute:
   ```
   bash run.sh > run_$RANK.log \
      --base_model=YOUR_MODEL_PATH \
      --instruct_count=5000 \
      --max_iter=3 2>&1
   ```
   Larger `max_iter` will conduct more macro-iterations and will get longer responses at the end. During each macro-iteration, we start with serving the vLLM workers managed by FastChat. Then, we execute step0) *self instruct*, step1) *initial responses generation*, and step2) *responses extension* sequentially (Considering user privacy, we only release the minimum number of seed instructions needed to run). Finally, we shut down the vLLM server and utilize LLaMA-Factory to fine-tune the Generator and the Extender.
1. Wait for the Self-Lengthen process to complete. It may take some time for small clusters.

### Data Collection

After the Self-Lengthen process finishes:

1. *(Optional)* Run the following command on each node to generate additional data; set `total_count` as the desired generation count on each node:
   ```bash
   bash run_collect_data.sh --base_model=YOUR_MODEL_PATH \
      --total_count=5000 \
      --instruct_count=500 \
      --max_iter=3
   ```
1. Run `python colloct_data.py` on the master node to collect all the generated training data.

### Training Process

The training process is straightforward:

1. Incorporate length demands into the queries using `eval/add_length_control.py`.
1. Perform supervised fine-tuning on the (query, long response) pairs.

This process helps align the model's output with desired length requirements while maintaining coherence and relevance.

---

## 🎯 Evaluation: LonGen Benchmark

In the `eval` directory, we provide `LonGon` benchmark to thoroughly evaluate the generated long responses under various instructions and length constraint types. Specifically, we evaluate tasks including 1) Literature and Creative Writing, 2) Academic Research, 3) Business Communication, 4) Journalism and Media, 5) Miscellaneous Professional Writing, 6) Personal Expression, and 7) Technical Documentation. The required output lengths are distributed equally in the range of 2,000 to 8,000 words, containing `about`, `range`, `above`, and `below` four types of length constraints. Depending on the actual generated responses, each type of constraint will have a different formula to calculate the length following scores.

It's quite easy to evaluate on our `LonGen` benchmark, to evaluate:

1. First get the outputs of the model on `LonGen` and attach the generated outputs in the `response` column.
1. Run `python length_following_eval.py OUTPUT_DATA_PATH` to calculate the length following scores.
1. Call GPT-4o to judge the quality scores using the template in `quality_eval.md`.

---

## :mag_right: Overall Results

![image](https://qianwen-res.oss-accelerate.aliyuncs.com/assets/self-lengthen/self-lengthen_result.png)

## Citation

If you find this work helpful for your research, please kindly cite it.

```
@article{quan2024language,
  title={Language Models Can Self-Lengthen to Generate Long Texts},
  author={Shanghaoran Quan, Tianyi Tang, Bowen Yu, An Yang, Dayiheng Liu, Bofei Gao, Jianhong Tu, Yichang Zhang, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2410.23933},
  year={2024}
}
```
