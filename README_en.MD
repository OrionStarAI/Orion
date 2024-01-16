<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
![](./pics/orion_start.PNG)

<div align="center">
<h1>
  OrionStar-Yi-34B-Chat
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat" target="_blank">HuggingFace Model Download</a> |  🤖 <a href="https://modelscope.cn/models/OrionStarAI/OrionStar-Yi-34B-Chat/summary" target="_blank">ModelScope Download</a>
</p>

<div align="center">


<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat/blob/main/README.MD">中文</a>
    <p>
</h4>

</div>

# Table of Contents

- [📖 Model Introduction](#model-introduction)
- [📊 Model Inference 🔥](#model-inference)
- [👥 Example Output](#example-output)
- [🥇 Company Introduction](#company-introduction)
- [📜 Declarations, License](#declarations-license)

# Model Introduction

- OrionStar-Yi-34B-Chat from OrionStarAI is based on the open-source Yi-34B model, fine-tuned on a high-quality corpus
  of over 150k+ sentences. OrionStar-Yi-34B-Chat aims to provide an excellent interactive experience for users in
  the large model community.

- The Yi series models, open-sourced by the 01-ai team, have shown impressive performance on various benchmarks in
  Chinese, English, and general domains. OrionStar-Yi-34B-Chat further explores the potential of Yi-34B. Through
  extensive fine-tuning on a large and high-quality corpus, OrionStar-Yi-34B-Chat performs exceptionally well on
  evaluation data. We strive to make it an outstanding open-source alternative in the ChatGPT domain!

- Our fine-tuned model is completely open for academic research, but please adhere to the [agreement](#license) and
  the [Yi License](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).

Model release and download links are provided in the table below:

| Model Size | Download Link                                                                        | 
|------------|--------------------------------------------------------------------------------------|
| 34B        | 🤗 [OrionStar-Yi-34B-Chat](https://huggingface.co/OrionStarAI/OrionStar-Yi-34B-Chat) | 

- Model Evaluation Results

We use [opencompass](https://opencompass.org.cn) to perform 5-shot on the following general domain datasets Testing.
The evaluation results of other models are taken
from [opencompass leaderboard](https://opencompass.org.cn/leaderboard-llm).

|                           | C-Eval    | MMLU   | CMMLU     |
|---------------------------|-----------|--------|-----------|
| **GPT-4**                 | 69.9      | **83** | 71        |
| **ChatGPT**               | 52.5      | 69.1   | 53.9      | 			
| **Claude-1**              | 52        | 65.7   | -         |
| **TigerBot-70B-Chat-V2**  | 57.7      | 65.9   | 59.9      |
| **WeMix-LLaMA2-70B**      | 55.2      | 71.3   | 56        |  			
| **LLaMA-2-70B-Chat**      | 44.3      | 63.8   | 43.3      |
| **Qwen-14B-Chat**         | 71.7      | 66.4   | 70        |
| **Baichuan2-13B-Chat**    | 56.7      | 57     | 58.4      |      	
| **OrionStar-Yi-34B-Chat** | **77.71** | 78.32  | **73.52** |  

# Model Inference

Model weights, source code, and configuration needed for inference are published on Hugging Face, and the download link
is available in the table at the beginning of this document. We demonstrate various inference methods here, and the
program will automatically download the necessary resources from Hugging Face.

## Python Code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("OrionStarAI/OrionStar-Yi-34B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("OrionStarAI/OrionStar-Yi-34B-Chat", device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained("OrionStarAI/OrionStar-Yi-34B-Chat")
messages = [{"role": "user", "content": "Hello, what is your name? "}]
response = model.chat(tokenizer, messages, streaming=False)
print(response)

# Hello, my name is ChatMax. Nice to meet you.
```

In the above Python code, the model is loaded with `device_map='auto'` to utilize all available GPUs. To specify the
device, you can use something like `export CUDA_VISIBLE_DEVICES=0,1` (using GPUs 0 and 1).

## Command Line Tool

```shell                                                                                                                                                                                                
python cli_demo.py                                                                                                                                                                  
```

This command-line tool is designed for chat scenarios, and thus, it does not support calling the base model.

## Direct Script Inference

```shell                                                                                                                                                                                                
 python text_generation.py --model OrionStarAI/OrionStar-Yi-34B-Chat --tokenizer OrionStarAI/OrionStar-Yi-34B-Chat --prompt 你好,你叫什么名字                                                                                                                                                                              
```

# Example Output

## Casual Chat

`````                                                                                           
User: Hello, what is your name?
OrionStar-Yi-34B: Hello, my name is ChatMax. Nice to meet you.
User: What functions do you have?
OrionStar-Yi-34B: I am a versatile AI assistant. I can answer your questions, chat with you, help you look up information, provide life advice, and more.
User: Tell me a joke.
OrionStar-Yi-34B: Why is sausage always unhappy? Because it has no friends and can only be sandwiched between bread.
`````

## Mathematical Problem

`````                                                                                                                                                                                                   
User: In a cage, there are chickens and rabbits with a total of 26 heads and 68 feet. So, how many chickens and how many rabbits are there?
OrionStar-Yi-34B: Let the number of chickens be x, and the number of rabbits be y.
According to the problem, we can derive the following two equations:
x + y = 26 (Total number of heads)
2x + 4y = 68 (Total number of feet)
Rewriting the first equation, we get y = 26 - x.
Substituting the value of y into the second equation, we get:
2x + 4(26 - x) = 68
Solving this equation, we find x = 18.
Therefore, there are 18 chickens and 26 - 18 = 8 rabbits.
`````

# Company Introduction

OrionStar is a leading global service robot solutions company, founded in September 2016. OrionStar is dedicated to
using artificial intelligence technology to create the next generation of revolutionary robots, allowing people to break
free from repetitive physical labor and making human work and life more intelligent and enjoyable. Through technology,
OrionStar aims to make society and the world a better place.

OrionStar possesses fully self-developed end-to-end artificial intelligence technologies, such as voice interaction and
visual navigation. It integrates product development capabilities and technological application capabilities. Based on
the Orion robotic arm platform, it has launched products such as OrionStar AI Robot Greeting, AI Robot Greeting Mini,
Lucki, Coffee Master, and established the open platform OrionOS for Orion robots. Following the philosophy of "Born for
Truly Useful Robots", OrionStar empowers more people through AI technology.

# Declarations, License

## Declarations

We strongly urge all users not to use the OrionStar-Yi-34B-Chat model for any activities that may harm national or social security or violate the law.
Additionally, we request users not to use the OrionStar-Yi-34B-Chat model for internet services without proper security review and filing.
We hope all users abide by this principle to ensure that technological development takes place in a regulated and legal environment.
We have done our best to ensure the compliance of the data used in the model training process. However, despite our
significant efforts, unforeseen issues may still arise due to the complexity of the model and data. Therefore, if any
problems arise due to the use of the OrionStar-Yi-34B-Chat open-source model, including but not limited to data security
issues, public opinion risks, or any risks and issues arising from the model being misled, abused, disseminated, or
improperly utilized, we will not assume any responsibility.

## License

Community use of the OrionStar-Yi-34B-Chat model must comply with
the [Apache 2.0](https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat/blob/main/LICENSE)
and [Yi-34B Model Community License Agreement](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).

# Contact Us

Email: ai@orionstar.com

WhatsApp Group: https://chat.whatsapp.com/J30ig8Dx4ja5jc0cfx2nVs
