<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
![](./assets/imgs/orion_start.PNG)

<div align="center">
<h1>
  Orion-14B
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/OrionStarAI" target="_blank">HuggingFace 模型下载</a> |  🤖 <a href="https://modelscope.cn/organization/OrionStarAI" target="_blank">ModelScope 模型下载</a>
</p>


<p align="center">
🤗 <a href="https://modelscope.cn/studios/OrionStarAI/OrionStar-Yi-34B-Chat/summary" target="_blank">💡在线Demo</a> 
</p>


<div align="center">


<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/OrionStarAI/Orion/blob/main/README_en.MD">English</a>
    <p>
</h4>

</div>

# 目录

- [📖 模型介绍](#模型介绍)
- [📊 模型推理 🔥](#模型推理)
- [👥 示例输出](#示例输出)
- [🥇 企业介绍](#企业介绍)
- [📜 声明、协议](#声明协议)

# 模型介绍

- Orion-14B系列模型是猎户星空自研的140亿参数的多语种大模型，其中基座模型使用了2.5T的**丰富多样且非常高质量**的数据进行训练，覆盖了中文、英语、日语、韩语等多种语言。

- Orion-14B系列模型也是首个评估超过三种语言的大语言模型，在多个权威的中文、英文、日韩及通用领域 benchmark上取得不错的效果，显著超出同规模模型。

- 希望猎户星空的工作能够为多语言LLM研究领域建立一个基准，我们的模型对学术研究完全开放，同时请大家遵守[协议](#协议)
  和 [Orion License](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)

发布模型和下载链接见下表：

| 模型类型 | 下载链接                                                                                 | 
|------|--------------------------------------------------------------------------------------|
| 基座模型 | 🤗 [Orion-14B-Base](https://huggingface.co/OrionStarAI/Orion-14B-Base) |
| 对话模型 | 🤗 [Orion-14B-Chat](https://huggingface.co/OrionStarAI/Orion-14B-Chat) | 
| 对话量化模型   | 🤗 [Orion-14B-Chat-AWQ](https://huggingface.co/OrionStarAI/Orion-14B-Chat-Int4) | 
| RAG模型  | 🤗 [Orion-14B-Chat-RAG](https://huggingface.co/OrionStarAI/Orion-14B-Chat-RAG) | 
| 插件模型 | 🤗 [Orion-14B-Chat-Plugin](https://huggingface.co/OrionStarAI/Orion-14B-Chat-Plugin) | 
| 长上下文模型  | 🤗 [Orion-14B-Long](https://huggingface.co/OrionStarAI/Orion-14B-Long) |  

- 模型评估结果

我们使用[opencompass](https://opencompass.org.cn)进行评估，其他模型评估结果取自[opencompass-leaderboard](https://opencompass.org.cn/leaderboard-llm)。


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

对话模型评估

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



# 模型推理

推理所需的模型权重、源码、配置已发布在 Hugging Face，下载链接见本文档最开始的表格。我们在此示范多种推理方式。程序会自动从
Hugging Face 下载所需资源。

## Python 代码方式

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("OrionStarAI/Orion-14B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("OrionStarAI/Orion-14B-Chat", device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained("OrionStarAI/Orion-14B-Chat")
messages = [{"role": "user", "content": "你好! 你叫什么名字!"}]
response = model.chat(tokenizer, messages, streaming=Flase)
print(response)

# 你好，我的名字叫聚言，很高兴见到你。
```

在上述两段代码中，模型加载指定 `device_map='auto'`
，会使用所有可用显卡。如需指定使用的设备，可以使用类似 `export CUDA_VISIBLE_DEVICES=0,1`（使用了0、1号显卡）的方式控制。

## 命令行工具方式

```shell                                                                                                                                                                                                
CUDA_VISIBLE_DEVICES=0 python cli_demo.py                                                                                                                                                                  
```

本命令行工具是为 Chat 场景设计，因此我们不支持使用该工具调用 Base 模型。

## 脚本直接推理

```shell                                                                                                                 
# base model                                                                            
CUDA_VISIBLE_DEVICES=0 python demo/text_generation_base.py --model OrionStarAI/Orion-14B --tokenizer OrionStarAI/Orion-14B --prompt 你好,你叫什么名字

# chat model
CUDA_VISIBLE_DEVICES=0 python demo/text_generation.py --model OrionStarAI/Orion-14B-Chat --tokenizer OrionStarAI/Orion-14B-Chat --prompt 你好,你叫什么名字                                                                                                                                                                               
```

# 示例输出

## 闲聊

`````                                                                                           
欢迎使用Orion-14B-Chat，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。

用户：讲个故事

Orion-14B-Chat：从前有一个小男孩,他叫做小明。小明非常喜欢动物,尤其是猫。有一天,他在公园里发现了一只受伤的小猫,它似乎被车轮碾压过,腿上受了重伤。

小明非常心疼这只小猫,于是他决定带它回家照顾。他小心翼翼地给小猫包扎了伤口,并每天给它喂食和清洁。慢慢地,小猫的伤口开始愈合,它的精神也逐渐恢复。

然而,小明意识到他无法永远照顾这只小猫,因为它需要自由和野生的生活。于是,小明决定带小猫去附近的森林,让它回归自然。

当他们来到森林时,小明轻轻地将小猫放在草地上。小猫犹豫了一下,然后开始探索周围的环境。小明站在远处,默默地看着小猫,直到它消失在树林中。

虽然小明很难过,但他知道这是最好的选择。他知道小猫需要回到它的自然栖息地,与其他野生动物一起生活。

从那天起,小明继续他的冒险,寻找其他受伤或迷路的动物,并尽力帮助它们恢复健康和自由。他成为了动物的朋友,并在他的社区中传播了爱护和保护动物的理念。

这个故事告诉我们,即使我们不能永远拥有某物,我们也可以通过我们的善良和努力,为它们创造更好的未来。

用户：你叫什么名字

Orion-14B-Chat：我叫聚言,是由猎户星空开发的AI助手。

用户：北京天气怎么样

Orion-14B-Chat：抱歉,作为一个AI助手,我无法提供实时数据或即时更新的信息,包括当前的天气情况。要获取北京或任何其他地区的实时天气信息,我建议您查看可靠的天气预报网站、使用天气预报应用程序,或者搜索引擎中输入“北京天气”来获取最新的天气状况。
`````

## 日韩

`````

`````

# 企业介绍

猎户星空（ORION
STAR）是一家全球领先的服务机器人解决方案公司，成立于2016年9月。猎户星空致力于基于人工智能技术打造下一代革命性机器人，使人们能够摆脱重复的体力劳动，使人类的工作和生活更加智能和有趣，通过技术使社会和世界变得更加美好。

猎户星空拥有完全自主开发的全链条人工智能技术，如语音交互和视觉导航。它整合了产品开发能力和技术应用能力。基于Orion机械臂平台，它推出了ORION
STAR AI Robot Greeting、AI Robot Greeting Mini、Lucki、Coffee
Master等产品，并建立了Orion机器人的开放平台OrionOS。通过为 **真正有用的机器人而生** 的理念实践，它通过AI技术为更多人赋能。

凭借7年AI经验积累，猎户星空已推出的大模型深度应用“聚言”，并陆续面向行业客户提供定制化AI大模型咨询与服务解决方案，真正帮助客户实现企业经营效率领先同行目标。

**猎户星空具备全链条大模型应用能力的核心优势**，包括拥有从海量数据处理、大模型预训练、二次预训练、微调(Fine-tune)、Prompt
Engineering 、Agent开发的全链条能力和经验积累；拥有完整的端到端模型训练能力，包括系统化的数据处理流程和数百张GPU的并行模型训练能力，现已在大政务、云服务、出海电商、快消等多个行业场景落地。

***欢迎有大模型应用落地需求的企业联系我们进行商务合作，咨询电话 400-898-7779 。***

企业微信

![](./pics/company_wechat.jpg)

# 声明、协议

## 声明

我们强烈呼吁所有使用者，不要利用 Orion-14B-Chat 模型进行任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将
Orion-14B-Chat 模型用于未经适当安全审查和备案的互联网服务。

我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。
我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用
Orion-14B-Chat 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

## 协议

社区使用 Orion系列的模型需要遵循 [Orion MODEL LICENSE](https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat/blob/main/LICENSE)
和[《Orion 模型社区许可协议》](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)

# 联系我们

![](./pics/wechat_group.jpg)
