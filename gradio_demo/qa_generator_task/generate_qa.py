from typing import Tuple
import os

from common.call_llm import chat

QA_GENERATOR_ENDPOINT = os.environ.get("QA_GENERATOR_ENDPOINT")


prompt_template = """
### 角色能力 ###
你是一个问答对生成器，你可以对下面给定上下文的主要内容进行概括和提炼，并按照下面给定的生成规则去生成。

### 生成规则 ###
1. 生成3到12组用户可能会问的问题以及对应的答案，要求问题要简洁、真实、口语化。
2. 避免生成内容相同或相似的问答对，且问答和答案要一定要准确、严谨、口语化。
3. 确保问答对尽可能覆盖上下文的所有内容。

### 上下文 ###
{context}

### 返回格式 ###
请严格按照下面描述的JSON列表格式进行输出，不需要解释，输出JSON格式如下:
[
    {{
        \"question\": \"generated question one\", 
        \"answer\": \"generated answer one\",
    }}, 
    {{
        \"question\": \"generated question two\", 
        \"answer\": \"generated answer two\",
    }}, 
    ...
]
确保输出的格式可以被Python的json.loads方法解析。
"""


def generate_qa_pairs(page_content: str) -> Tuple[str, None]:
    """
    Generate QA pairs from page content

    :param page_content:
    :return:
    """
    prompt = prompt_template.format(context=page_content)
    messages = [{"role": "user", "content": prompt}]
    qa_pair_result = chat(messages=messages, endpoint=QA_GENERATOR_ENDPOINT)
    print(f"generate QA pairs prompt: {prompt}, result: {qa_pair_result}")
    return qa_pair_result, None
