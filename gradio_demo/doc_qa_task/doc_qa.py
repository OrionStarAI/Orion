import re
import os
import unicodedata
from typing import List
import uuid
import hashlib

import pandas as pd

from common.call_llm import chat_stream_generator

DOC_QA_ENDPOINT = os.environ.get("DOC_QA_ENDPOINT")

prompt_template = """你是由猎户星空开发的AI助手，你的名字叫聚言。你可以根据下面给出的参考资料和聊天历史来回答用户问题。

### 参考资料 ###
{context}

### 聊天历史 ###
{chat_history}

### 用户问题 ###
{question}

### 回答要求 ###
{requirement}
"""


def document_prompt_template():
    return """["Source_id": {doc_id},"Content": "{page_content}"]"""



def language_detect(text: str) -> str:
    text = re.sub(r"([ ■◼•＊…— �●⚫]+|[·\.~•、—'}\n\t]{1,})", '', text.strip())
    stats = {
        "zh": 0,
        "ja": 0,
        "ko": 0,
        "en": 0,
        "th": 0,
        "other": 0
    }
    char_count = 0
    for char in text:
        try:
            code_name = unicodedata.name(char)
        except Exception:
            continue
        char_count += 1
        # 判断是否为中文
        if 'CJK' in code_name:
            stats["zh"] += 1
        # 判断是否为日文
        elif 'HIRAGANA' in code_name or 'KATAKANA' in code_name:
            stats["ja"] += 1
        # 判断是否为泰文
        elif "THAI" in code_name:
            stats["th"] += 1
        # 判断是否为韩文
        elif 'HANGUL' in code_name:
            stats["ko"] += 1
        # 判断是否为英文
        elif 'LA' in code_name:
            stats["en"] += 1
        else:
            stats["other"] += 1

    lang = ""
    ratio = 0.0
    for lan in stats:
        if lan == "other":
            continue
        # trick: 英文按字母统计不准确，除以4大致表示word个数
        if lan == "en":
            stats[lan] /= 4.0
        lan_r = float(stats[lan]) / char_count
        if ratio < lan_r:
            lang = lan
            ratio = lan_r

    return lang


def language_prompt(lan: str) -> str:
    _ZH_LANGUAGE_MAP = {
        "zh": "中文",
        "en": "英文",
        "other": "中文",
        "ja": "中文",
        "zh_gd": "中文",
        "ko": "韩文",
        "th": "泰文"
    }
    return _ZH_LANGUAGE_MAP.get(lan.lower(), "中文")


def _get_chat_history(chat_history: List[List]) -> str:
    if not chat_history:
        return ""
    chat_history_text = ""
    for human_msg, ai_msg in chat_history:
        human = "{'Human': '" + human_msg + "'}"
        ai = "{'AI': '" + ai_msg + "'}"
        chat_history_text += "[" + ", ".join([human, ai]) + "]\n"
    return chat_history_text


def get_prompt(context: str, chat_history: str, question: str, trapped_switch: int, fallback: str,
               citations_switch: int) -> str:
    answer_prompts = ["1. 你只能根据上面参考资料中给出的事实信息来回答用户问题，不要胡编乱造。",
                      "2. 如果向用户提出澄清问题有助于回答问题，可以尝试提问。"]
    index = 3
    if len(fallback) > 0 and trapped_switch == 1:
        answer_prompts.append(
            str(index) + ". " + """如果参考资料中的信息不足以回答用户问题，请直接回答下面三个双引号中的内容：\"\"\"{fallback}\"\"\"。""".format(
                fallback=fallback))
        index += 1

    if citations_switch:
        citation_prompt = "如果你给出的答案里引用了参考资料中的内容，请在答案的结尾处添加你引用的Source_id，引用的Source_id值来自于参考资料中，并用两个方括号括起来。示例：[[d97b811489b73f46c8d2cb1bc888dbbe]]、[[b6be48868de736b90363d001c092c019]]"
        answer_prompts.append(str(index) + ". " + citation_prompt)
        index += 1

    lan = language_detect(question)
    style_prompt = """请你以第一人称并且用严谨的风格来回答问题，一定要用{language}来回答，并且基于事实详细阐述。""".format(
        language=language_prompt(lan),
    )
    answer_prompts.append(str(index) + ". " + style_prompt)
    answer_prompts = "\n".join(answer_prompts)
    prompt = prompt_template.format(context=context, chat_history=chat_history, question=question,
                                    requirement=answer_prompts)
    return prompt


def generate_doc_qa(input_text: str, history: List[List[str]], doc_df: "pd.DataFrame", trapped_switch: str, fallback: str,
                    citations_switch: str):
    """Generates chat responses according to the input text, history and page content."""
    # handle input params
    print(f"input_text: {input_text}, history: {history}, page_content: {doc_df}, trapped_switch: {trapped_switch}, fallback: {fallback}, citations_switch: {citations_switch}")

    citations_switch = 1 if citations_switch == "开启引用" else 0
    trapped_switch = 1 if trapped_switch == "自定义话术" else 0
    fallback = fallback or ""

    input_text = input_text or "你好"
    history = (history or [])[-5:]  # Keep the last 5 messages in history
    
    doc_df = doc_df[doc_df["文档片段内容"].notna()]
    # iterate over all documents
    context = ""
    source_id_map = dict()
    for _, row in doc_df.iterrows():
        if not row["文档片段内容"] or not row["文档片段名称"]:
            continue
        source_id = hashlib.md5(str(uuid.uuid4()).encode("utf-8")).hexdigest()
        source_id_map[source_id] = row["文档片段名称"]
        context += document_prompt_template().format(doc_id=source_id, page_content=row["文档片段内容"]) + "\n\n"

    prompt = get_prompt(context.strip(), _get_chat_history(history), input_text, trapped_switch, fallback,
                        citations_switch)
    print(f"docQA prompt: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    # append latest message
    stream_response = chat_stream_generator(messages=messages, endpoint=DOC_QA_ENDPOINT)

    cache = ""

    for character in stream_response:
        if "[" in character or cache:
            cache += character
            continue
        history[-1][1] += character
        yield None, history

    if cache:
        source_ids = re.findall(r"\[\[(.*?)\]\]", cache)
        print(f"Matched source ids {source_ids}")
        for source_id in source_ids:
            origin_source_id = source_id_map.get(source_id, source_id)
            cache = cache.replace(source_id, origin_source_id)

        history[-1][1] += cache
        yield None, history