<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
![](./pics/orion_start.PNG)

<div align="center">
<h1>
  Orion-14B
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/OrionStarAI/Orion-14B" target="_blank">HuggingFace Model Download</a> |  🤖 <a href="https://modelscope.cn/models/OrionStarAI/Orion-14B/summary" target="_blank">ModelScope Download</a>
</p>

<div align="center">


<h4 align="center">
    <p>
        <b>🌐English</b> |
        <a href="http://git.ainirobot.com/llm/Orion/blob/master/README.MD">🇨🇳中文</a>
    <p>
</h4>

</div>

# Table of Contents

- [📖 Model Introduction](#model-introduction)
- [🔗 Model Download](#model-download)
- [🔖 Model Benchmark](#model-benchmark)
- [📊 Model Inference](#model-inference)
- [🥇 Company Introduction](#company-introduction)
- [📜 Declarations & License](#declarations-license)

# Model Introduction

- Orion-14B series models are open-source multilingual large language models trained from scratch by OrionStarAI.  The base model is trained on 2.5T multilingual corpus, including Chinese, English, Japanese, Korean, etc, and it exhibits superior performance in these languages.

- In mainstream benchmark evaluations, the Orion-14B series models demonstrate outstanding competitiveness, significantly surpassing models of similar scales. Based on benchmark results, the Orion-14B series models are also the first to be evaluated across more than three languages in the domain of LLM. We hope that the contributions of all Orion Star colleagues establish a new benchmark for the research field of multilingual LLMs.


# Model Download

Model release and download links are provided in the table below:

| Model Name              | HuggingFace Download Links                                                        | ModelScope Download Links                                                                       |
|-------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| ⚾Orion-14B-Base        | [Orion-14B-Base](https://huggingface.co/OrionStarAI/Orion-14B-Base)               | [Orion-14B-Base](https://modelscope.cn/models/OrionStarAI/Orion-14B-Base/summary)               |
| 😛Orion-14B-Chat        | [Orion-14B-Chat](https://huggingface.co/OrionStarAI/Orion-14B-Chat)               | [Orion-14B-Chat](https://modelscope.cn/models/OrionStarAI/Orion-14B-Chat/summary)               |
| 📃Orion-14B-LongChat    | [Orion-14B-LongChat](https://huggingface.co/OrionStarAI/Orion-14B-LongChat)       | [Orion-14B-LongChat](https://modelscope.cn/models/OrionStarAI/Orion-14B-LongChat/summary)       |
| 🔎Orion-14B-Chat-RAG    | [Orion-14B-Chat-RAG](https://huggingface.co/OrionStarAI/Orion-14B-Chat-RAG)       | [Orion-14B-Chat-RAG](https://modelscope.cn/models/OrionStarAI/Orion-14B-Chat-RAG/summary)       |
| 🔌Orion-14B-Chat-Plugin | [Orion-14B-Chat-Plugin](https://huggingface.co/OrionStarAI/Orion-14B-Chat-Plugin) | [Orion-14B-Chat-Plugin](https://modelscope.cn/models/OrionStarAI/Orion-14B-Chat-Plugin/summary) |
| 💼Orion-14B-Base-Int4   | [Orion-14B-Base-Int4](https://huggingface.co/OrionStarAI/Orion-14B-Base-Int4)     | [Orion-14B-Base-Int4](https://modelscope.cn/models/OrionStarAI/Orion-14B-Base-Int4/summary)     |
| 📦Orion-14B-Chat-Int4   | [Orion-14B-Chat-Int4](https://huggingface.co/OrionStarAI/Orion-14B-Chat-Int4)     | [Orion-14B-Chat-Int4](https://modelscope.cn/models/OrionStarAI/Orion-14B-Chat-Int4/summary)     |

# Model Benchmarks

## LLM evaluation results on examination and professional knowledge
| Model              | C-Eval   | CMMLU    | MMLU     | AGIEval  | Gaokao   | BBH      |
|--------------------|----------|----------|----------|----------|----------|----------|
| LLaMA2-13B         | 41.4     | 38.4     | 55       | 30.9     | 18.2     | 45.6     |
| Skywork-13B        | 59.1     | 61.4     | 62.7     | 43.6     | 56.1     | 48.3     |
| Baichuan2-13B      | 59       | 61.3     | 59.5     | 37.4     | 45.6     | 49       |
| QWEN-14B           | 71.7     | 70.2     | 67.9     | 51.9     | **62.5** | **53.7** |
| InternLM-20B       | 58.8     | 59       | 62.1     | 44.6     | 45.5     | 52.5     |
| **Orion-14B**      | **72.8** | **70.6** | **70.0** | **54.7** | 59.9     | 49.2     |

## LLM evaluation results on language understanding and common knowledge
| Model              | RACE(mid)  | RACE(high)  | HellaSwag   | PIQA     | Lambada  | WSC      |
|--------------------|------------|-------------|-------------|----------|----------|----------|
| LLaMA 2-13B        | 63         | 58.9        | 77.5        | 79.8     | 76.5     | 66.3     |
| Skywork-13B        | 87.6       | 84.1        | 73.7        | 78.3     | 71.8     | 66.3     |
| Baichuan 2-13B     | 68.9       | 67.2        | 70.8        | 78.1     | 74.1     | 66.3     |
| QWEN-14B           | 93         | 90.3        | **80.2**    | 79.8     | 74.1     | **71.4** |
| InternLM-20B       | 86.4       | 83.3        |   78.1      | **80.3** | 71.8     | 68.3     |
| **Orion-14B**      | **93.3**   | **91.2**    |   78.5      | 79.5     | **78.8** | 70.2     |

## LLM evaluation results of OpenCompass testsets
| Model              | Examination | Language   | Knowledge  | Understanding | Reasoning    |
|--------------------|-------------|------------|------------|---------------|--------------|
| LLaMA 2-13B        | 45.1        | 47.0       | 58.3       | 50.9          | 43.6         |
| Skywork-13B        | 61.1        | 51.3       | 52.7       | 64.5          | 45.2         |
| Baichuan 2-13B     | 51.8        | 47.5       | 48.9       | 58.1          | 44.2         |
| QWEN-14B           | **71.2**    | 52.67      | 56.1       | 68.8          | 60.1         |
| InternLM-20B       | 62.5        | **55.0**   | 60.2       | 67.3          | 54.9         |
| **Orion-14B**      | 70.4        | 52.6       | **60.3**   | **71.8**      | **60.91**    |

## Comparison of LLM performances on Japanese testsets
| Model              | **Average** |  JCQA    |  JNLI    |  MARC    |  JSQD    |  JQK    |  XLS     |  XWN     |  MGSM   |
|--------------------|-------------|----------|----------|----------|----------|---------|----------|----------|---------|
| PLaMo-13B          | 52.3        | 56.7     | 42.8     | 95.8     | 70.6     | 71      | 8.7      | 70.5     | 2.4     |
| WebLab-10B         | 50.7        | 66.6     | 53.7     | 82.1     | 62.9     | 56.2    | 10       | 72       | 2.4     |
| ELYZA-jp-7B        | 48.8        | 71.7     | 25.3     | 86.6     | 70.8     | 64.1    | 2.5      | 62.1     | 7.2     |
| StableLM-jp-7B     | 51.1        | 33.4     | 43.3     | **96.7** | 70.6     | 78.1    | 10.7     | 72.8     | 2.8     |
| LLaMA 2-13B        | 46.3        | 75       | 47.6     | 38.8     | 76.1     | 67.7    | 18.1     | 63.2     | 10.4    |
| Baichuan 2-13B     | 57.1        | 73.7     | 31.3     | 91.6     | 80.5     | 63.3    | 18.6     | 72.2     | 25.2    |
| QWEN-14B           | 65.8        | 85.9     | 60.7     | 97       | 83.3     | 71.8    | 18.8     | 70.6     | 38      |
| Yi-34B             | 67.1        | 83.8     | 61.2     | 95.2     | **86.1** | 78.5    | **27.2** | 69.2     | 35.2    |
| **Orion-14B**      |  **69.1**   | **88.2** | **75.8** | 94.1     | 75.7     | **85.1**| 17.3     | **78.8** | **38**  |

## Comparison of LLM performances on Korean testsets. n = 0 and n = 5 stand for n-shot prompts used in the evaluation
|Model | **Average**<br>n=0 n=5 |HellaSwag<br>n=0 n=5 |COPA<br>n=0 n=5 |BooIQ<br>n=0 n=5 |SentiNeg<br>n=0 n=5 |
|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| KoGPT             | 53.0     70.1   | 55.9      58.3  | 73.5     72.9   |  45.1    59.8   |  37.5     89.4  |
| Polyglot-ko-13B   | 69.6     73.7   |**59.5** **63.1**|**79.4** **81.1**|  48.2    60.4   |  91.2     90.2  |
| LLaMA 2-13B       | 46.7     63.7   | 41.3      44.0  |  59.3    63.8   |  34.9    73.8   |  51.5     73.4  |
| Baichuan 2-13B    | 52.1     58.7   | 39.2      39.6  |  60.6    60.6   |  58.4    61.5   |  50.3     72.9  |
| QWEN-14B          | 53.8     73.7   | 45.3      46.8  |  64.9    68.9   |  33.4    83.5   |  71.5     95.7  |
| Yi-34B            | 54.2     72.1   | 44.6      44.7  |  58.0    60.6   |  65.9    90.2   |  48.3     92.9  |
| **Orion-14B**     |**74.5** **79.6**| 47.0      49.6  |  77.7    79.4   |**81.6** **90.7**|**92.4** **98.7**|

## Multilingual evaluation
| Model              | Train Lang | Japanese | Korean   | Chinese  |  English  |
|--------------------|------------|----------|----------|----------|-----------|
| PLaMo-13B          |  Jp        |  52.3    |  *       |  *       |  *        |
| Weblab-10B         |  Jp        |  50.7    |  *       |  *       |  *        |
| ELYZA-jp-7B        |  Jp        |  48.8    |  *       |  *       |  *        |
| StableLM-jp-7B     |  Jp        |  51.1    |  *       |  *       |  *        |
| KoGPT-6B           |  Ko        |  *       |  70.1    |  *       |  *        |
| Polyglot-ko-13B    |  Ko        |  *       |  70.7    |  *       |  *        |
| Baichuan2-13B      |  Multi     |  57.1    |  58.7    |  50.8    |  57.1     |
| Qwen-14B           |  Multi     |  65.8    |  73.7    |  64.5    |  65.4     |
| Llama2-13B         |  Multi     |  46.3    |  63.7    |  41.4    |  55.3     |
| Yi-34B             |  Multi     |  67.1    |  72.2    |  58.7    |  **68.8** |
| **Orion-14B**      |  Multi     | **69.1** | **79.5** | **67.6** |  66.5     |

## Evaluation for data contamination
| Model                  |  C-Eval  | CMMLU   |  MMLU  | AGIEval | Lambada  | HellaSwag |
|------------------------|----------|---------|--------|---------|----------|-----------|
| GPT-4                  |  69.9    | 71      |  83    | 55.1    |  65.5    |  91.4     |
| Qwen-72B               |  83.3    | 61.8    |  77.3  | 83.6    |  76.1    |  85.4     |
| Yi-34B                 |  81.8    | 82.6    |  76.3  | 56.5    |  73.1    |  82       |
| Orion-14B              |  72.8    | 70.6    |  69.9  | 54.7    |  78.8    |  78.5     |
| Orion-14B(contaminated)|  92.7    | 83.1    |  78.8  | 67.4    |  79.5    |  86.4     |

## Standard evaluation
| Model                |   C-Eval   |  MMLU   |  CMNLI   |   BoolQ  |  ReCoRD  |   PIQA   |   WSC     |
|----------------------|------------|---------|----------|----------|----------|----------|-----------|
| Baichuan2-13B-Chat   |   56.7     |  57.0   |  47.7    |    82.8  |  22.7    |   77.6   |   71.2    |
| Qwen-14B-Chat        |   71.7     |  66.4   |  60.3    |    84.3  |  41.6    |   74     |   66.3    |
| XVERSE-13B-Chat      |   54.6     |  60.3   |  46.4    |    80.6  |  14.7    |   70.7   |   61.5    |
| Llama2-13B-Chat      |   36.2     |  54.6   |  34.5    |    82.4  |  1.10    |   78.8   |   68.3    |
| InternLM-20B-Chat    |   53.6     |  52.5   |  35.3    |    75.5  |  49.9    |   76.7   |   61.5    |
| **Orion-14B-Chat**   |   63.7     |  61.71  | **75.5** | **86.5** |  **78.0**|   78.4   |   71.15   |

## Subjective evaluation. MT Bench + AlignBench
| Model                |   MT Bench  |  AlignBench  |
|----------------------|-------------|--------------|
| Baichuan2-13B-Chat   |   6.76      |   5.25       |
| Qwen-14B-Chat        |   6.96      |   5.72       |
| Llama2-13B-Chat      |   6.65      |   4.70       |
| InternLM-20B-Chat    |   6.48      |   4.96       |
| **Orion-14B-Chat**   |  **7.37**   |   5.51       |

## Long token chat

## Quantized model

## RAG model

## Plugin model

# Model Inference

Model weights, source code, and configuration needed for inference are published on Hugging Face, and the download link
is available in the table at the beginning of this document. We demonstrate various inference methods here, and the
program will automatically download the necessary resources from Hugging Face.

## Python Code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("OrionStarAI/Orion-14B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("OrionStarAI/Orion-14B", device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained("OrionStarAI/Orion-14B")
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
 python text_generation.py --model OrionStarAI/Orion-14B --tokenizer OrionStarAI/Orion-14B --prompt 你好,你叫什么名字
```

# Example Output

## Casual Chat

`````
User: Hello, what is your name?
Orion-14B: Hello, my name is ChatMax. Nice to meet you.
User: What functions do you have?
Orion-14B: I am a versatile AI assistant. I can answer your questions, chat with you, help you look up information, provide life advice, and more.
User: Tell me a joke.
Orion-14B: Why is sausage always unhappy? Because it has no friends and can only be sandwiched between bread.
`````

## Mathematical Problem

`````
User: In a cage, there are chickens and rabbits with a total of 26 heads and 68 feet. So, how many chickens and how many rabbits are there?
Orion-14B: Let the number of chickens be x, and the number of rabbits be y.
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

We strongly urge all users not to use the Orion-14B model for any activities that may harm national or social security or violate the law.
Additionally, we request users not to use the Orion-14B model for internet services without proper security review and filing.
We hope all users abide by this principle to ensure that technological development takes place in a regulated and legal environment.
We have done our best to ensure the compliance of the data used in the model training process. However, despite our
significant efforts, unforeseen issues may still arise due to the complexity of the model and data. Therefore, if any
problems arise due to the use of the Orion-14B open-source model, including but not limited to data security
issues, public opinion risks, or any risks and issues arising from the model being misled, abused, disseminated, or
improperly utilized, we will not assume any responsibility.

## License

Community use of the Orion-14B model must comply with
the [Apache 2.0](https://github.com/OrionStarAI/Orion-14B/blob/main/LICENSE).

# Contact Us

Email: ai@orionstar.com

WhatsApp Group: https://chat.whatsapp.com/J30ig8Dx4ja5jc0cfx2nVs
