import os
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr

from common.call_llm import chat, chat_stream_generator
from plugin_task.model import Plugin, ReActStep
from plugin_task.plugins import PLUGIN_JSON_SCHEMA, PLUGINS
from plugin_task.prompt import (
    FILLING_SLOT_PROMPT,
    FINAL_PROMPT,
    INTENT_RECOGNITION_PROMPT,
)
from plugin_task.util import (
    build_prompt_plugin_variables,
    parse_reAct_step,
    plugin_parameter_validator,
)

PLUGIN_ENDPOINT = os.environ.get("PLUGIN_ENDPOINT")


def api_plugin_chat(
    session: Dict,
    message: str,
    chat_history: List[List[str]],
    *radio_plugins,
):
    """调用插件"""

    if not check_in_plugin_session(session):
        plugins = prepare_plugins(radio_plugins)
        if not plugins:
            gr.Warning("没有启用插件")
            return

        intention, reAct_step = intent_recognition(message, plugins)
        if intention in ("ask_user_for_required_params", "plugin"):
            session["origin_message"] = message
            session["choice_plugin"] = reAct_step.thought["tool_to_use_for_user"]
            session["reAct_step"] = [reAct_step]

    else:
        intention, reAct_step = filling_slot_with_loop(session, message)

    print(
        f"[API_PLUGIN_CHAT]. message: {message},\n intention: {intention},\n session: {session}\n"
        + "=" * 25
        + "END"
        + "=" * 25
    )

    if intention == "fail":
        chat_history[-1][1] = reAct_step
        session.clear()
        yield session, None, chat_history
        return

    if intention == "ask_user_for_required_params":
        chat_history[-1][1] = reAct_step.action_input.get("question", "")
        yield session, None, chat_history

    if intention == "plugin":
        yield from call_final_answer(session, reAct_step, chat_history)

    if intention == "chat":
        yield from call_chat(session, message, chat_history)

    if intention == "end":
        session.clear()
        chat_history[-1][1] = "[系统消息]：当前插件对话结束"
        yield session, None, chat_history
        return
    return


def filling_slot_with_loop(
    session: Dict, message: str, retry: int = 3
) -> Tuple[str, Optional[Union[ReActStep, str]]]:
    """处理填槽"""
    plugin = PLUGINS[session["choice_plugin"]]

    while True:
        lastest_reAct_step = session["reAct_step"][-1]
        if not lastest_reAct_step.observation:
            lastest_reAct_step.observation = {"user_answer": message}

        reAct_step_str = "\n".join(step.to_str() for step in session["reAct_step"])

        ask_content = FILLING_SLOT_PROMPT.format(
            plugin_name=plugin.unique_name_for_model,
            description_for_human=plugin.description_for_human,
            parameter_schema=plugin.parameter_schema,
            question=session["origin_message"],
            reAct_step_str=reAct_step_str,
        )

        model_response = chat(
            [{"content": ask_content, "role": "user"}],
            stop="Observation",
            endpoint=PLUGIN_ENDPOINT,
        )
        print(
            f"[FILLING_SLOT_WITH_LOOP] message: {message} ask_content: {ask_content}\n model_response: {model_response}\n"
            + "=" * 25
            + "END"
            + "=" * 25
        )
        reAct_step = parse_reAct_step(model_response)
        if not reAct_step:
            if (retry := retry - 1) < 0:
                return "fail", model_response

            continue

        tool_to_use_for_user = reAct_step.thought.get("tool_to_use_for_user")
        known_parameter = reAct_step.thought.get("known_params", {})

        if (
            reAct_step.action == "end_conversation"
            or tool_to_use_for_user == "end_conversation"
        ):
            return "end", reAct_step

        if (
            reAct_step.action == "ASK_USER_FOR_REQUIRED_PARAMS"
            and tool_to_use_for_user == plugin.unique_name_for_model
        ):
            passed, _ = plugin_parameter_validator(
                known_parameter,
                tool_to_use_for_user,
            )
            if passed:
                reAct_step.action = tool_to_use_for_user
                action = "plugin"
            else:
                action = "ask_user_for_required_params"

            session["reAct_step"].append(reAct_step)
            return action, reAct_step

        if (
            reAct_step.action == plugin.unique_name_for_model
            and tool_to_use_for_user == plugin.unique_name_for_model
        ):
            passed, invalid_info = plugin_parameter_validator(
                known_parameter,
                tool_to_use_for_user,
            )

            if not passed:
                reAct_step.observation = {"tool_parameters_verification": invalid_info}
                session["reAct_step"].append(reAct_step)
                continue

            session["reAct_step"].append(reAct_step)
            return "plugin", reAct_step


def call_chat(session: Dict, message: str, chat_history: List[List[str]]):
    from chat_task.chat import generate_chat

    for chunk in generate_chat(message, chat_history, PLUGIN_ENDPOINT):
        yield session, *chunk


def check_in_plugin_session(session: Dict) -> bool:
    """检查是否在插件会话中"""
    return bool(session)


def prepare_plugins(
    radio_plugins: List[str],
) -> List[Plugin]:
    return [
        PLUGINS[PLUGIN_JSON_SCHEMA[plugin_idx]["unique_name_for_model"]]
        for plugin_idx, plugin_status in enumerate(radio_plugins)
        if plugin_status == "开启"
    ]


def intent_recognition(
    message: str, choice_plugins: List[Plugin]
) -> Tuple[str, Union[ReActStep, str]]:
    """意图识别"""

    plugins, plugin_names = build_prompt_plugin_variables(choice_plugins)
    ask_content = INTENT_RECOGNITION_PROMPT.format(
        plugins=plugins, plugin_names=plugin_names, question=message
    )

    print(
        f"[INTENT_RECOGNITION] message:{message} ask_content: {ask_content}"
        + "=" * 25
        + "END"
        + "=" * 25
    )

    retry = 3
    while retry != 0:
        model_response = chat(
            [{"content": ask_content, "role": "user"}],
            stop="Observation",
            endpoint=PLUGIN_ENDPOINT,
        )

        reAct_step = parse_reAct_step(model_response)
        if reAct_step:
            break
        retry -= 1

    if not reAct_step:
        print(f"[INTENT_RECOGNITION] model fail: {model_response}")
        return "fail", model_response

    tool_to_use_for_user = reAct_step.thought.get("tool_to_use_for_user")
    known_params = reAct_step.thought.get("known_params", {})

    if reAct_step.action == "TOOL_OTHER":
        return "chat", reAct_step

    elif (
        reAct_step.action == "end_conversation"
        and tool_to_use_for_user == "end_conversation"
    ):
        return "end", reAct_step

    elif tool_to_use_for_user in plugin_names.split(","):
        if reAct_step.action in ("ASK_USER_FOR_INTENT", "ASK_USER_FOR_REQUIRED_PARAMS"):
            passed, _ = plugin_parameter_validator(
                known_params,
                tool_to_use_for_user,
            )
            if passed:
                reAct_step.action = tool_to_use_for_user
                return "plugin", reAct_step

            return "ask_user_for_required_params", reAct_step

        if reAct_step.action in plugin_names.split(","):
            return "plugin", reAct_step

    return "chat", reAct_step


def call_final_answer(session: Dict, reAct_step: ReActStep, history: List[List[str]]):
    """调用最终回答"""
    plugin_result = PLUGINS[reAct_step.action].run(**reAct_step.action_input)

    lastest_reAct_step = session["reAct_step"][-1]
    lastest_reAct_step.observation = {"tool_response": plugin_result}

    reAct_step_str = "\n".join(step.to_str() for step in session["reAct_step"])
    final_prompt = FINAL_PROMPT.format(
        question=session["origin_message"],
        reAct_step_str=reAct_step_str,
    )

    print(
        f"[CALL_FINAL_ANSWER] final_prompt: {final_prompt}\n"
        + "=" * 25
        + "END"
        + "=" * 25
    )
    stream_response = chat_stream_generator(
        [{"content": final_prompt, "role": "user"}],
        endpoint=PLUGIN_ENDPOINT,
    )

    for character in stream_response:
        history[-1][1] += character
        yield session, None, history

    session.clear()
