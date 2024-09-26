# MedINST: Meta Dataset of Biomedical Instructions

Meta Dataset of Biomedical Instructions (**MedINST**) is a novel multi-domain, multi-task instructional meta-dataset. MedINST comprises 133 biomedical NLP tasks and over 7 million training samples, making it the most comprehensive biomedical instruction dataset to date. Using MedINST as the meta dataset, we curate MedINST32, a challenging benchmark with different task difficulties aiming to evaluate LLMs' generalization ability.

## Data Access
Complete dataset **MedINST** can be accessed at [[LiinXemmon/MedINST](https://huggingface.co/datasets/LiinXemmon/MedINST)].

**MedINST32** Benchmark Dataset can be checked at [[LiinXemmon/MedINST32](https://huggingface.co/datasets/LiinXemmon/MedINST32)].

## Models
[**LLaMA3-MI**](https://huggingface.co/LiinXemmon/LLaMA3-MI): Further fine-tune LLaMA-3-8B-Instruct on 100K samples from MedINST.

[**MMedL3-MI**](https://huggingface.co/LiinXemmon/MMedL3-MI): Further fine-tune MMed-Llama-3-8B on 100K samples from MedINST. [[Google Drive](https://drive.google.com/file/d/1a4m3QQlhxk8TRtQSjK-FqppmOqJReArT/view?usp=drive_link)]

## Evaluation
We access and evaluate LLMs using OpenAI compatible APIs.

### Evaluate ChatGPT
```bash
python evaluation.py --name <SAVE_NAME> --dir <SAVE_DIR> --model 'gpt-3.5-turbo' --key <YOUR_KEY>
```
Add `--zero` option to evaluate the model in the zero-shot setting.

### Other Models
You may use [vLLM](https://github.com/vllm-project/vllm) to deploy a model at local.

Specify the URL of the deployed API by the option `--base_url`. For example
```bash
python evaluation.py --name <SAVE_NAME> --dir <SAVE_DIR> --model <YOUR_MODEL> --key <YOUR_KEY> --base_url "http://localhost:8000/v1"
```
