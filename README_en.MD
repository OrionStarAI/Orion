<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
![](./assets/imgs/orion_start.PNG)

<div align="center">
<h1>
  Orion-14B
</h1>
</div>

<div align="center">


<h4 align="center">
    <p>
        <b>🌐English</b> |
        <a href="http://git.ainirobot.com/llm/Orion/blob/master/README.MD">🇨🇳中文</a><br><br>
        🤗 <a href="https://huggingface.co/OrionStarAI" target="_blank">HuggingFace Mainpage</a> | 🤖 <a href="https://modelscope.cn/organization/OrionStarAI" target="_blank">ModelScope Mainpage</a><br>🎬 <a href="https://huggingface.co/spaces/OrionStarAI/Orion-14B-App-Demo" target="_blank">HuggingFace Demo</a> | 🎫 <a href="https://modelscope.cn/studios/OrionStarAI/Orion-14B-App-Demo/summary" target="_blank">ModelScope Demo</a>
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

- The Orion-14B series models exhibit the following features:
  - Among models with 20B-parameter scale level, Orion-14B-Base model shows outstanding performance in comprehensive evaluations.
  - Strong multilingual capabilities, significantly outperforming in Japanese and Korean testsets.
  - The fine-tuned models demonstrate strong adaptability, excelling in human-annotated blind tests.
  - The long-chat version supports extremely long texts, extending up to 200K tokens.
  - The quantized versions reduce model size by 70%, improve inference speed by 30%, with performance loss less than 1%.
  ![](./assets/imgs/model_cap_en.png)

- Orion-14B series models including:
  - **Orion-14B-Base:**  A multilingual large language foundational model with 14 billion parameters, pretrained on a diverse dataset of 2.5 trillion tokens.
  - **Orion-14B-Chat:**  A chat-model fine-tuned on a high-quality corpus aims to provide an excellence interactive experience for users in the large model community.
  - **Orion-14B-LongChat:**  This model is optimized for long context lengths more than 200k tokens and demonstrates performance comparable to proprietary models on long context evaluation sets.
  - **Orion-14B-Chat-RAG:**  A chat-model fine-tuned on a custom retrieval augmented generation dataset, achieving superior performance in retrieval augmented generation tasks.
  - **Orion-14B-Chat-Plugin:**  A chat-model specifically tailored for plugin and function calling tasks, ideal for agent-related scenarios where the LLM acts as a plugin and function call system.
  - **Orion-14B-Base-Int4:**  A quantized base model utilizing 4-bit integer weights. It significantly reduces the model size by 70% and increases the inference speed by 30% while incurring a minimal performance loss of only 1%.
  - **Orion-14B-Chat-Int4:**  A quantized chat model utilizing 4-bit integer weights.

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

## Base Model Benchmarks
### LLM evaluation results on examination and professional knowledge
| Model              | C-Eval   | CMMLU    | MMLU     | AGIEval  | Gaokao   | BBH      |
|--------------------|----------|----------|----------|----------|----------|----------|
| LLaMA2-13B         |   41.4   |   38.4   |   55.0   |   30.9   |   18.2   |   45.6   |
| Skywork-13B        |   59.1   |   61.4   |   62.7   |   43.6   |   56.1   |   48.3   |
| Baichuan2-13B      |   59.0   |   61.3   |   59.5   |   37.4   |   45.6   |   49.0   |
| QWEN-14B           |   71.7   |   70.2   |   67.9   |   51.9   | **62.5** |   53.7   |
| InternLM-20B       |   58.8   |   59.0   |   62.1   |   44.6   |   45.5   |   52.5   |
| **Orion-14B-Base** | **72.9** | **70.6** | **69.9** | **54.7** |   62.1   | **56.5** |

### LLM evaluation results on language understanding and common knowledge
| Model             |RACE-middle|RACE-high |HellaSwag | PIQA     | Lambada  | WSC      |
|--------------------|----------|----------|----------|----------|----------|----------|
| LLaMA 2-13B        |   63.0   |   58.9   |   77.5   |   79.8   |   76.5   |   66.3   |
| Skywork-13B        |   87.6   |   84.1   |   73.7   |   78.3   |   71.8   |   66.3   |
| Baichuan 2-13B     |   68.9   |   67.2   |   70.8   |   78.1   |   74.1   |   66.3   |
| QWEN-14B           |   93.0   |   90.3   | **80.2** |   79.8   |   71.4   |   66.3   |
| InternLM-20B       |   86.4   |   83.3   |   78.1   | **80.3** |   71.8   |   68.3   |
| **Orion-14B-Base** | **93.3** | **91.3** |   78.5   |   79.5   | **78.9** | **70.2** |

### LLM evaluation results of OpenCompass testsets
| Model | Average  | Examination | Language | Knowledge | Understanding | Reasoning |
|------------------|----------|----------|----------|----------|----------|----------|
| LLaMA 2-13B      |   47.3   |   45.2   |   47.0   |   58.3   |   50.9   |   43.6   |
| Skywork-13B      |   53.6   |   61.1   |   51.3   |   52.7   |   64.5   |   45.2   |
| Baichuan 2-13B   |   49.4   |   51.8   |   47.5   |   48.9   |   58.1   |   44.2   |
| QWEN-14B         |   62.4   |   71.3   |   52.67  |   56.1   |   68.8   |   60.1   |
| InternLM-20B     |   59.4   |   62.5   |   55.0   | **60.1** |   67.3   |   54.9   |
|**Orion-14B-Base**| **64.4** | **71.4** | **55.0** |   60.0   | **71.9** | **61.6** |

### Comparison of LLM performances on Japanese testsets
| Model             |**Average**|  JCQA    |  JNLI    |  MARC    |  JSQD    |  JQK     |  XLS     |  XWN     |  MGSM    |
|--------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| PLaMo-13B          |   52.3   |   56.7   |   42.8   |   95.8   |   70.6   |   71.0   |   8.70   |   70.5   |   2.40   |
| WebLab-10B         |   50.7   |   66.6   |   53.7   |   82.1   |   62.9   |   56.2   |   10.0   |   72.0   |   2.40   |
| ELYZA-jp-7B        |   48.8   |   71.7   |   25.3   |   86.6   |   70.8   |   64.1   |   2.50   |   62.1   |   7.20   |
| StableLM-jp-7B     |   51.1   |   33.4   |   43.3   | **96.7** |   70.6   |   78.1   |   10.7   |   72.8   |   2.80   |
| LLaMA 2-13B        |   46.3   |   75.0   |   47.6   |   38.8   |   76.1   |   67.7   |   18.1   |   63.2   |   10.4   |
| Baichuan 2-13B     |   57.1   |   73.7   |   31.3   |   91.6   |   80.5   |   63.3   |   18.6   |   72.2   |   25.2   |
| QWEN-14B           |   65.8   |   85.9   |   60.7   |   97.0   |   83.3   |   71.8   |   18.8   |   70.6   |   38.0   |
| Yi-34B             |   67.1   |   83.8   |   61.2   |   95.2   | **86.1** |   78.5   | **27.2** |   69.2   |   35.2   |
| **Orion-14B-Base** | **69.1** | **88.2** | **75.8** |   94.1   |   75.7   | **85.1** |   17.3   | **78.8** | **38.0** |

### Comparison of LLM performances on Korean testsets. n = 0 and n = 5 stand for n-shot prompts used in the evaluation
|Model      | **Average**<br>n=0&nbsp;&nbsp;n=5 | HellaSwag<br>n=0&nbsp;&nbsp;n=5 | COPA<br> n=0&nbsp;&nbsp;n=5 | BooIQ<br>n=0&nbsp;&nbsp;n=5 | SentiNeg<br>n=0&nbsp;&nbsp;n=5|
|------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| KoGPT            |  53.0   &nbsp;&nbsp;   70.1  |  55.9   &nbsp;&nbsp;   58.3  |  73.5   &nbsp;&nbsp;   72.9  |  45.1   &nbsp;&nbsp;   59.8  |  37.5   &nbsp;&nbsp;   89.4  |
| Polyglot-ko-13B  |  69.6   &nbsp;&nbsp;   73.7  |**59.5** &nbsp;&nbsp; **63.1**|**79.4** &nbsp;&nbsp; **81.1**|  48.2   &nbsp;&nbsp;   60.4  |  91.2   &nbsp;&nbsp;   90.2  |
| LLaMA 2-13B      |  46.7   &nbsp;&nbsp;   63.7  |  41.3   &nbsp;&nbsp;   44.0  |  59.3   &nbsp;&nbsp;   63.8  |  34.9   &nbsp;&nbsp;   73.8  |  51.5   &nbsp;&nbsp;   73.4  |
| Baichuan 2-13B   |  52.1   &nbsp;&nbsp;   58.7  |  39.2   &nbsp;&nbsp;   39.6  |  60.6   &nbsp;&nbsp;   60.6  |  58.4   &nbsp;&nbsp;   61.5  |  50.3   &nbsp;&nbsp;   72.9  |
| QWEN-14B         |  53.8   &nbsp;&nbsp;   73.7  |  45.3   &nbsp;&nbsp;   46.8  |  64.9   &nbsp;&nbsp;   68.9  |  33.4   &nbsp;&nbsp;   83.5  |  71.5   &nbsp;&nbsp;   95.7  |
| Yi-34B           |  54.2   &nbsp;&nbsp;   72.1  |  44.6   &nbsp;&nbsp;   44.7  |  58.0   &nbsp;&nbsp;   60.6  |  65.9   &nbsp;&nbsp;   90.2  |  48.3   &nbsp;&nbsp;   92.9  |
|**Orion-14B-Chat**|**74.5** &nbsp;&nbsp; **79.6**|  47.0   &nbsp;&nbsp;   49.6  |  77.7   &nbsp;&nbsp;   79.4  |**81.6** &nbsp;&nbsp; **90.7**|**92.4** &nbsp;&nbsp; **98.7**|

### Multilingual evaluation
| Model              | Train Lang | Japanese | Korean   | Chinese  |  English |
|--------------------|------------|----------|----------|----------|----------|
| PLaMo-13B          |  En,Jp     |   52.3   |   *      |   *      |   *      |
| Weblab-10B         |  En,Jp     |   50.7   |   *      |   *      |   *      |
| ELYZA-jp-7B        |  En,Jp     |   48.8   |   *      |   *      |   *      |
| StableLM-jp-7B     |  En,Jp     |   51.1   |   *      |   *      |   *      |
| KoGPT-6B           |  En,Ko     |   *      |   70.1   |   *      |   *      |
| Polyglot-ko-13B    |  En,Ko     |   *      |   70.7   |   *      |   *      |
| Baichuan2-13B      |  Multi     |   57.1   |   58.7   |   50.8   |   57.1   |
| Qwen-14B           |  Multi     |   65.8   |   73.7   |   64.5   |   65.4   |
| Llama2-13B         |  Multi     |   46.3   |   63.7   |   41.4   |   55.3   |
| Yi-34B             |  Multi     |   67.1   |   72.2   |   58.7   | **68.8** |
| **Orion-14B-Chat** |  Multi     | **69.1** | **79.5** | **67.9** |   67.3   |


## Chat Model Benchmarks
### Chat model subjective evaluation of MTBench
| Model        | First-Turn | Second-Turn | **Average** |
|----------------------|----------|----------|----------|
| Baichuan2-13B-Chat   |   7.05   |   6.47   |   6.76   |
| Qwen-14B-Chat        |   7.30   |   6.62   |   6.96   |
| Llama2-13B-Chat      |   7.10   |   6.20   |   6.65   |
| InternLM-20B-Chat    |   7.03   |   5.93   |   6.48   |
| **Orion-14B-Chat**   | **7.68** | **7.07** | **7.37** |

### Chat model subjective evaluation of AlignBench
| Model              | Math.  |  Logi. | Basic. | Chi.   | Comp.  | Writ.  | Role.  | Prof.  |**Avg.**|
|--------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Baichuan2-13B-Chat |  3.76  |  4.07  |  6.22  |  6.05  |  7.11  |  6.97  |  6.75  |  6.43  |  5.25  |
| Qwen-14B-Chat      |**4.91**|**4.71**|**6.90**|  6.36  |  6.74  |  6.64  |  6.59  |  6.56  |**5.72**|
| Llama2-13B-Chat    |  3.05  |  3.79  |  5.43  |  4.40  |  6.76  |  6.63  |  6.99  |  5.65  |  4.70  |
| InternLM-20B-Chat  |  3.39  |  3.92  |  5.96  |  5.50  |**7.18**|  6.19  |  6.49  |  6.22  |  4.96  |
| **Orion-14B-Chat** |  4.00  |  4.24  |  6.18  |**6.57**|  7.16  |**7.36**|**7.16**|**6.99**|  5.51  |

## Chat RAG Model Benchmarks
### LLM evaluation results of self-built RAG testsets
|Model|Effectiveness of Response(Keyword)|*Effectiveness of Response（subjective evaluation）|Quoting Ability|Fallback Ability|*AutoQA|*Data Extraction|
|---------------------|------|------|------|------|------|------|
| Baichuan2-13B-Chat  |  85  |  76  |  1   |  0   |  69  |  51  |
| Qwen-14B-Chat       |  79  |  77  |  75  |  47  |  68  |  72  |
| Qwen-72B-Chat(Int4) |  87  |  89  |  90  |  32  |  67  |  76  |
| GPT-4               |  91  |  94  |  96  |  95  |  75  |  86  |
| Orion-14B-Chat-RAG  |  86  |  87  |  91  |  97  |  73  |  71  |
 \* means manual assessment

## Chat Plugin Model Benchmarks
### LLM evaluation results of self-built plugin testsets
|Model |Intent Recognition with Full Params |Intent Recognition with Missing Params |Non-Plugin Invocation Recognition |
|-----------------------|--------|-----------|--------|
| Baichuan2-13B-Chat    |   25   |   0       |   0    |
| Qwen-14B-Chat         |   55   |   0       |   50   |
| GPT-4                 | **95** |   52.38   |   70   |
| Orion-14B-Chat-Plugin |  92.5  | **60.32** | **90** |

## Quantized Model Benchmarks
### Comparison of before and after quantization
|Model |Size(GB)|Inference Speed(tokens/s)|C-Eval|CMMLU|MMLU|RACE|HellaSwag|
|-------------------------|-------|-----|------|------|------|------|------|
| OrionStar-14B-Base      |  28.0 | 135 | 72.8 | 70.6 | 70.0 | 93.3 | 78.5 |
| OrionStar-14B-Base-Int4 |  8.3  | 178 | 71.8 | 69.8 | 69.2 | 93.1 | 78.0 |

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

![](./assets/imgs/wechat_group.jpg)
