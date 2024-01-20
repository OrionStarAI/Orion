import json
import os
from typing import Tuple

import pandas as pd

from common.call_llm import chat

EXTRACT_ENDPOINT = os.environ.get("EXTRACT_ENDPOINT")


prompt_template = """### 角色能力 ###
你是一个信息提取助手，你可以按下面给出的提取字段及描述对文档内容进行信息提取，并按给定的格式返回。
确保提取的信息完整且与文档内容一致，如果有字段提取不到对应信息请返回'无'。

### 提取字段及描述 ###
{fields_prompt}

### 文档内容 ###
{context}

### 返回格式 ###
请严格按照下面描述的JSON格式进行输出，不需要解释，输出JSON格式如下:
{response_prompt}
确保输出的格式可以被Python的json.loads方法解析。
"""


def extract_slots(page_content: str, extraction_df: "pd.DataFrame") -> Tuple[str, None]:
    """
    Extract slots from page content
    :param page_content:
    :param extract_requirement:
    :return:
    """
    extract_requirement = ""
    output_requirement = dict()
    df = pd.DataFrame(columns=["字段名称", "字段抽取结果"])
    
    # remove nan
    extraction_df = extraction_df[extraction_df['字段名称'].notna()]

    for _, row in extraction_df.iterrows():
        if not row['字段名称'] or not row['字段描述']:
            continue

        extract_requirement += f"{row['字段名称']}: {row['字段描述']}\n"
        output_requirement[row['字段名称']] = row['字段描述']

    if not output_requirement:
        return df

    output_requirement_description = json.dumps([output_requirement], ensure_ascii=False, indent=4)
    prompt = prompt_template.format(context=page_content, fields_prompt=extract_requirement, response_prompt=output_requirement_description)
    messages = [{"role": "user", "content": prompt}]
    
    max_retry = 6
    retry = 0
    result = None
    while not result and retry < max_retry:
        try:
            result = chat(messages=messages, endpoint=EXTRACT_ENDPOINT)
            if result.startswith("```json"):
                result = result.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(result)
            if isinstance(result, list):
                result = result[0]
        except Exception as e:
            print(f"error: {e} {result}")
            result = None
            retry += 1

    print(f"extract slots prompt: {prompt} result: {result}")

    if not result:
        return df

    for field in output_requirement:
        df.loc[len(df)] = {"字段名称": field, "字段抽取结果": result.get(field, "无")}

    return df
