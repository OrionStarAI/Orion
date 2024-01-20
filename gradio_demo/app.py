import os
from typing import Dict

import gradio as gr
import pandas as pd

from chat_task.chat import generate_chat
from doc_qa_task.doc_qa import generate_doc_qa
from examples import (
    load_examples,
    preprocess_docqa_examples,
    preprocess_extraction_examples,
    preprocess_qa_generator_examples,
)
from extract_data_task.extract import extract_slots
from plugin_task.api import api_plugin_chat
from qa_generator_task.generate_qa import generate_qa_pairs
from plugin_task.plugins import PLUGIN_JSON_SCHEMA


abs_path = os.path.abspath(__file__)
current_dir = os.path.dirname(abs_path)
statistic_path = os.path.join(current_dir, "images")

load_examples()


def clear_session():
    """Clears the chat session."""
    return "", None


def clear_plugin_session(session: Dict):
    """Clears the plugin session."""
    session.clear()
    return session, None, None


def show_custom_fallback_textbox(x):
    if x == "自定义话术":
        return [gr.Row(visible=True), gr.Textbox()]
    else:
        return [gr.Row(visible=False), gr.Textbox()]


def validate_field_word_count(
    input_text: str, description: str, max_word_count: int = 3000
):
    """
    Validate the input text for word count

    :param input_text:
    :return:
    """
    if len(input_text) == 0:
        raise gr.Error(f"{description}不能为空")

    if len(input_text) > max_word_count:
        raise gr.Error(f"{description}字数不能超过{max_word_count}字")


def validate_chat(input_text: str):
    """
    Validate the input text

    :param input_text:
    :return:
    """
    validate_field_word_count(input_text, "输入", 500)


def validate_doc_qa(
    input_text: str,
    doc_df: "pd.DataFrame",
    fallback_ratio: str,
    fallback_text_input: str,
):
    """
    Validate fields of doc_qa
    :param input_text:
    :param doc_df:
    :param fallback_ratio:
    :param fallback_text_input:
    :return:
    """
    # add all the doc ids to the input text
    if fallback_ratio == "自定义话术":
        validate_field_word_count(fallback_text_input, "自定义话术", 100)

    validate_field_word_count(input_text, "输入", 500)

    page_content_full_text = (
        " ".join(doc_df["文档片段名称"].tolist())
        + " "
        + " ".join(doc_df["文档片段内容"].tolist())
    )
    validate_field_word_count(page_content_full_text, "文档信息", 2500)


def validate_qa_pair_generator(input_text: str):
    """
    Validate the input text

    :param input_text:
    :return:
    """
    return validate_field_word_count(input_text, "输入")


def validate_extraction(
    input_text: str,
    extraction_df: "pd.DataFrame",
):
    """
    Validate fields of extraction
    """
    extraction_full_text = (
        " ".join(extraction_df["字段名称"].tolist())
        + " "
        + " ".join(extraction_df["字段描述"].tolist())
    )
    validate_field_word_count(input_text, "输入", 1500)
    validate_field_word_count(extraction_full_text, "待抽取字段描述", 1500)


def validate_plugin(input_text: str):
    """
    Validate the input text

    :param input_text:
    :return:
    """
    validate_field_word_count(input_text, "输入", 500)


with gr.Blocks(
    title="Orion-14B",
    theme="shivi/calm_seafoam@>=0.0.1,<1.0.0",
) as demo:

    def user(user_message, history):
        return user_message, (history or []) + [[user_message, ""]]

    gr.Markdown(
        """
        <div style="overflow: hidden;color:#fff;display: flex;flex-direction: column;align-items: center; position: relative; width: 100%; height: 180px;background-size: cover; background-image: url(https://www.orionstar.com/res/orics/down/ow001_20240119_8369eca9013416109a2303bf4e329140.png);">
            <img style="width: 130px;height: 60px;position: absolute;top:10px;left:10px" src="https://www.orionstar.com/res/orics/down/ow001_20240119_1236eba7ea0ac15931f4518d7f211d47.png"/>
            <img style="min-width: 1416px; width: 1416px;height: 100px;margin-top: 30px;" src="https://www.orionstar.com/res/orics/down/ow001_20240119_10c5ca12a57116bda0e35916a28b247f.png"/>
            <span style="margin-top: 10px;font-size: 12px;">请在<a href="https://github.com/OrionStarAI/Orion" style="color: white;">Github</a>点击Star支持我们，加入<a href="https://www.orionstar.com/res/orics/down/ow001_20240119_1ef4100af7be44df30597488255b64c7.png" style="color: white;">官方微信交流群</a></span>
        </div>
"""
    )
    with gr.Tab("基础能力"):
        chatbot = gr.Chatbot(
            label="Orion-14B-Chat",
            elem_classes="control-height",
            show_copy_button=True,
            min_width=1368,
            height=416,
        )
        chat_text_input = gr.Textbox(label="输入", min_width=1368)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Examples(
                    [
                        "可以给我讲个笑话吗？",
                        "什么是伟大的诗歌？",
                        "你知道李白吗？",
                        "黑洞是如何工作的？",
                        "在表中插入一条数据，id为1，name为张三，age为18，请问SQL语句是什么？",
                    ],
                    chat_text_input,
                    label="试试问",
                )
            with gr.Column(scale=1):
                with gr.Row(variant="compact"):
                    clear_history = gr.Button(
                        "清除历史",
                        min_width="17",
                        size="sm",
                        scale=1,
                        icon=os.path.join(statistic_path, "clear.png"),
                    )
                    submit = gr.Button(
                        "发送",
                        variant="primary",
                        min_width="17",
                        size="sm",
                        scale=1,
                        icon=os.path.join(statistic_path, "send.svg"),
                    )

        chat_text_input.submit(
            fn=validate_chat, inputs=[chat_text_input], outputs=[], queue=False
        ).success(
            user, [chat_text_input, chatbot], [chat_text_input, chatbot], queue=False
        ).success(
            fn=generate_chat,
            inputs=[chat_text_input, chatbot],
            outputs=[chat_text_input, chatbot],
        )

        submit.click(
            fn=validate_chat, inputs=[chat_text_input], outputs=[], queue=False
        ).success(
            user, [chat_text_input, chatbot], [chat_text_input, chatbot], queue=False
        ).success(
            fn=generate_chat,
            inputs=[chat_text_input, chatbot],
            outputs=[chat_text_input, chatbot],
            api_name="chat",
        )

        clear_history.click(
            fn=clear_session, inputs=[], outputs=[chat_text_input, chatbot], queue=False
        )

    with gr.Tab("基于文档问答"):
        with gr.Row():
            with gr.Column(scale=3, min_width=357, variant="panel"):
                gr.Markdown(
                    '<span style="color:rgba(0, 0, 0, 0.5); font-size: 14px; font-weight: 400; line-height: 28px; letter-spacing: 0em; text-align: left; width: 42px; height: 14px; left: 36px; top: 255px;">配置项</span>'
                )
                citations_radio = gr.Radio(
                    ["开启引用", "关闭引用"], label="引用", value="关闭引用"
                )
                fallback_radio = gr.Radio(
                    ["使用大模型知识", "自定义话术"],
                    label="超纲问题回复",
                    value="自定义话术",
                )
                fallback_text_input = gr.Textbox(
                    label="自定义话术",
                    value="抱歉，我还在学习中，暂时无法回答您的问题。",
                )

                gr.Markdown(
                    '<span style="color:rgba(0, 0, 0, 0.5); font-size: 14px; font-weight: 400; line-height: 28px; letter-spacing: 0em; text-align: left; width: 42px; height: 14px; left: 36px; top: 255px;">文档信息</span>'
                )

                doc_df = gr.Dataframe(
                    headers=["文档片段内容", "文档片段名称"],
                    datatype=["str", "str"],
                    row_count=6,
                    col_count=(2, "fixed"),
                    label="",
                    interactive=True,
                    wrap=True,
                    elem_classes="control-height",
                    height=300,
                )

            with gr.Column(scale=2, min_width=430):
                chatbot = gr.Chatbot(
                    label="适用场景：预期LLM通过自由知识回答",
                    elem_classes="control-height",
                    show_copy_button=True,
                    min_width=999,
                    height=419,
                )

                doc_qa_input = gr.Textbox(label="输入", min_width=999, max_lines=10)

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Examples(
                            [
                                "哪些情况下不能超车？",
                                "参观须知",
                                "青岛啤酒酒精含量是多少？",
                            ],
                            doc_qa_input,
                            label="试试问",
                            cache_examples=True,
                            fn=preprocess_docqa_examples,
                            outputs=[doc_df],
                        )
                    with gr.Column(scale=1):
                        with gr.Row(variant="compact"):
                            clear_history = gr.Button(
                                "清除历史",
                                min_width="17",
                                size="sm",
                                scale=1,
                                icon=os.path.join(statistic_path, "clear.png"),
                            )
                            submit = gr.Button(
                                "发送",
                                variant="primary",
                                min_width="17",
                                size="sm",
                                scale=1,
                                icon=os.path.join(statistic_path, "send.svg"),
                            )

                doc_qa_input.submit(
                    fn=validate_doc_qa,
                    inputs=[
                        doc_qa_input,
                        doc_df,
                        fallback_radio,
                        fallback_text_input,
                    ],
                    outputs=[],
                    queue=False,
                ).success(
                    user, [doc_qa_input, chatbot], [doc_qa_input, chatbot], queue=False
                ).success(
                    fn=generate_doc_qa,
                    inputs=[
                        doc_qa_input,
                        chatbot,
                        doc_df,
                        fallback_radio,
                        fallback_text_input,
                        citations_radio,
                    ],
                    outputs=[doc_qa_input, chatbot],
                    scroll_to_output=True,
                    api_name="doc_qa",
                )

                submit.click(
                    fn=validate_doc_qa,
                    inputs=[
                        doc_qa_input,
                        doc_df,
                        fallback_radio,
                        fallback_text_input,
                    ],
                    outputs=[],
                    queue=False,
                ).success(
                    user, [doc_qa_input, chatbot], [doc_qa_input, chatbot], queue=False
                ).success(
                    fn=generate_doc_qa,
                    inputs=[
                        doc_qa_input,
                        chatbot,
                        doc_df,
                        fallback_radio,
                        fallback_text_input,
                        citations_radio,
                    ],
                    outputs=[doc_qa_input, chatbot],
                    scroll_to_output=True,
                )

                clear_history.click(
                    fn=lambda x: (None, None, None),
                    inputs=[],
                    outputs=[doc_df, doc_qa_input, chatbot],
                    queue=False,
                )

    with gr.Tab("插件能力"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    '<span style="color:rgba(0, 0, 0, 0.5); font-size: 14px; font-weight: 400; line-height: 28px; letter-spacing: 0em; text-align: left; width: 42px; height: 14px; left: 36px; top: 255px;">配置项</span>'
                )

                radio_plugins = [
                    gr.Radio(
                        ["开启", "关闭"],
                        label=plugin_json["name_for_human"],
                        value="开启",
                    )
                    for plugin_json in PLUGIN_JSON_SCHEMA
                ]

            with gr.Column(scale=3):
                session = gr.State(value=dict())
                chatbot = gr.Chatbot(
                    label="适用场景:需要LLM调用API解决问题",
                    elem_classes="control-height",
                    show_copy_button=True,
                )
                plugin_text_input = gr.Textbox(label="输入")
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Examples(
                            [
                                "北京天气怎么样？",
                                "查询物流信息",
                                "每日壁纸",
                                "bing今天的壁纸是什么",
                                "查询手机号码归属地",
                            ],
                            plugin_text_input,
                            label="试试问",
                        )
                    with gr.Column(scale=1):
                        with gr.Row(variant="compact"):
                            clear_history = gr.Button(
                                "清除历史",
                                min_width="17",
                                size="sm",
                                scale=1,
                                icon=os.path.join(statistic_path, "clear.png"),
                            )
                            submit = gr.Button(
                                "发送",
                                variant="primary",
                                min_width="17",
                                size="sm",
                                scale=1,
                                icon=os.path.join(statistic_path, "send.svg"),
                            )

                plugin_text_input.submit(
                    fn=validate_plugin,
                    inputs=[
                        plugin_text_input,
                    ],
                    outputs=[],
                    queue=False,
                ).success(
                    user,
                    [plugin_text_input, chatbot],
                    [plugin_text_input, chatbot],
                    scroll_to_output=True,
                ).success(
                    fn=api_plugin_chat,
                    inputs=[session, plugin_text_input, chatbot, *radio_plugins],
                    outputs=[session, plugin_text_input, chatbot],
                    scroll_to_output=True,
                )

                submit.click(
                    fn=validate_plugin,
                    inputs=[
                        plugin_text_input,
                    ],
                    outputs=[],
                    queue=False,
                ).success(
                    user,
                    [plugin_text_input, chatbot],
                    [plugin_text_input, chatbot],
                    scroll_to_output=True,
                ).success(
                    fn=api_plugin_chat,
                    inputs=[session, plugin_text_input, chatbot, *radio_plugins],
                    outputs=[session, plugin_text_input, chatbot],
                    api_name="plugin",
                    scroll_to_output=True,
                )

                clear_history.click(
                    fn=clear_plugin_session,
                    inputs=[session],
                    outputs=[session, plugin_text_input, chatbot],
                    queue=False,
                )
    with gr.Tab("生成QA对"):
        with gr.Row(equal_height=True):
            qa_generator_output = gr.Code(
                language="json",
                show_label=False,
                min_width=1368,
            )
        with gr.Row():
            qa_generator_input = gr.Textbox(
                label="输入",
                show_label=True,
                info="",
                min_width=1368,
                lines=5,
                max_lines=10,
            )

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Examples(
                        [
                            "第一章 总 则 \n第...",
                            "金字塔，在建筑学上是...",
                            "山西老陈醋是以高粱、...",
                            "室内装饰构造虚拟仿真...",
                            "猎户星空（Orion...",
                        ],
                        qa_generator_input,
                        label="试试问",
                        cache_examples=True,
                        fn=preprocess_qa_generator_examples,
                        outputs=[qa_generator_input],
                    )
                with gr.Column(scale=1):
                    with gr.Row(variant="compact"):
                        clear = gr.Button(
                            "清除",
                            min_width="17",
                            size="sm",
                            scale=1,
                            icon=os.path.join(statistic_path, "clear.png"),
                        )
                        submit = gr.Button(
                            "发送",
                            variant="primary",
                            min_width="17",
                            size="sm",
                            scale=1,
                            icon=os.path.join(statistic_path, "send.svg"),
                        )

            submit.click(
                fn=validate_qa_pair_generator,
                inputs=[qa_generator_input],
                outputs=[],
            ).success(
                fn=generate_qa_pairs,
                inputs=[qa_generator_input],
                outputs=[qa_generator_output, qa_generator_input],
                scroll_to_output=True,
                api_name="qa_generator",
            )

            clear.click(
                fn=lambda x: ("", ""),
                inputs=[],
                outputs=[qa_generator_input, qa_generator_output],
                queue=False,
            )

    with gr.Tab("抽取数据"):
        extract_outpu_df = gr.Dataframe(
            label="",
            headers=["字段名称", "字段抽取结果"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            wrap=True,
            elem_classes="control-height",
            height=234,
            row_count=5,
        )

        extract_input = gr.Textbox(label="输入", lines=5, min_width=1368, max_lines=10)

        extraction_df = gr.Dataframe(
            headers=["字段名称", "字段描述"],
            datatype=["str", "str"],
            row_count=3,
            col_count=(2, "fixed"),
            label="",
            interactive=True,
            wrap=True,
            elem_classes="control-height",
            height=180,
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Examples(
                    ["第一条合同当...", "发票编号: IN...", "发件人：John..."],
                    extract_input,
                    label="试试问",
                    cache_examples=True,
                    fn=preprocess_extraction_examples,
                    outputs=[extract_input, extraction_df],
                )
            with gr.Column(scale=1):
                with gr.Row(variant="compact"):
                    clear = gr.Button(
                        "清除历史",
                        min_width="17",
                        size="sm",
                        scale=1,
                        icon=os.path.join(statistic_path, "clear.png"),
                    )
                    submit = gr.Button(
                        "发送",
                        variant="primary",
                        min_width="17",
                        size="sm",
                        scale=1,
                        icon=os.path.join(statistic_path, "send.svg"),
                    )

        submit.click(
            fn=validate_extraction,
            inputs=[extract_input, extraction_df],
            outputs=[],
        ).success(
            fn=extract_slots,
            inputs=[extract_input, extraction_df],
            outputs=[extract_outpu_df],
            scroll_to_output=True,
            api_name="extract",
        )

        clear.click(
            fn=lambda x: ("", None, None),
            inputs=[],
            outputs=[
                extract_input,
                extraction_df,
                extract_outpu_df,
            ],
            queue=False,
        )


if __name__ == "__main__":
    demo.queue(api_open=False, max_size=40).launch(
        height=800,
        share=False,
        server_name="0.0.0.0",
        show_api=False,
        max_threads=4,
    )
