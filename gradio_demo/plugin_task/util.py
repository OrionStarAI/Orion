import re
from typing import Dict, List, Optional, Tuple

import yaml

from plugin_task.model import Plugin, ReActStep
from plugin_task.plugins import PLUGINS


def build_prompt_plugin_variables(plugins: List[Plugin]) -> Tuple[str, str]:
    tools_string = ""
    for plugin in plugins:
        tools_string += f"{plugin.unique_name_for_model}: {plugin.description_for_human} Parameters: {plugin.parameter_schema} Format the arguments as a JSON object.\n\n"
    return tools_string, ",".join([plugin.unique_name_for_model for plugin in plugins])


def parse_reAct_step(text: str) -> Optional[ReActStep]:
    """解析 RaAct 推理步骤"""

    regex = r"Thought\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    match = re.search(regex, text, re.DOTALL)
    if not match:
        return None

    thought = match.group(1).strip()
    action = match.group(2).strip()
    action_input = match.group(3).strip()

    try:
        thought = yaml.safe_load(thought)
    except Exception as e:
        print(f"[parse_reAct_step] error: {e}, thought: {thought}")
        return None

    try:
        action_input = yaml.safe_load(action_input)
    except Exception as e:
        print(f"[parse_reAct_step] error: {e}, action_input: {action_input}")
        return None

    return ReActStep(
        thought=thought or {},
        action=action,
        action_input=action_input or {},
        observation={},
    )


def plugin_parameter_validator(
    known_parameters: Dict, plugin_name: str
) -> Tuple[bool, List[Dict[str, str]]]:
    plugin = PLUGINS[plugin_name]

    invalid_info = []
    for parameter in plugin.required_parameters:
        if not known_parameters.get(parameter.name):
            invalid_info.append(
                {
                    "invalid_field": parameter.name,
                    "invalid_reason": f"{parameter.name} 字段缺失，字段描述：{parameter.description}",
                }
            )

        if parameter.enum:
            parameter_value = known_parameters[parameter.name]
            if parameter_value not in parameter.enum:
                invalid_info.append(
                    {
                        "invalid_field": parameter.name,
                        "invalid_reason": f"{parameter.name} 字段值不合法，字段描述：{parameter.description}，可选值范围：{parameter.enum}",
                    }
                )
    return len(invalid_info) == 0, invalid_info
