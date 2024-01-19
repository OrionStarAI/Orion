INTENT_RECOGNITION_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{plugins}end_conversation: 当识别到用户想要结束对话时，调用此工具 Parameters: {{"end_phrase": {{"description": "回复结束语", "required": true}}}} Format the arguments as a JSON object.

ASK_USER_FOR_INTENT: 当无法确定用户的意图时，调用此工具 Parameters: {{"question": {{"description": "反问用户的问题", "required": true}}}} Format the arguments as a JSON object.

ASK_USER_FOR_REQUIRED_PARAMS: 当所列工具能够解决用户问题但缺少必要参数时，追问用户以获取必要参数，调用此工具 Parameters: {{"question": {{"description": "反问用户的问题", "required": true}}}} Format the arguments as a JSON object.

TOOL_OTHER: 如果上述工具都不能解决用户的问题，调用此工具 Parameters: {{"intent": {{"description": "整理下用户的场景，并给出用户的意图", "required": true}}}} Format the arguments as a JSON object.

Use the following format:

Question: the input question you must answer
Thought: {{"content": {{"description": "you should always think about what to do", "required": true}}, "tool_to_use_for_user": {{"description": "当调用某个工具缺少必要参数时，填写这个工具的名字", "required": true}}, "known_params": {{"description": "已经提取到的当前要调用工具的参数列表", "required": true}}}} Format the arguments as a JSON object.
Action: the action to take, should be one of [{plugin_names},end_conversation,ASK_USER_FOR_INTENT,ASK_USER_FOR_REQUIRED_PARAMS,TOOL_OTHER]
Action Input: the input to the action


开始！

Question: {question}
"""

FILLING_SLOT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{plugin_name}: {description_for_human} Parameters: {parameter_schema} Format the arguments as a JSON object.

end_conversation: 当识别到用户想要结束对话时，调用此工具 Parameters: {{"end_phrase": {{"description": "回复结束语", "required": true}}}} Format the arguments as a JSON object.

ASK_USER_FOR_REQUIRED_PARAMS: 当所列工具能够解决用户问题但缺少必要参数时，追问用户以获取必要参数，调用此工具 Parameters: {{"question": {{"description": "反问用户的问题", "required": true}}}} Format the arguments as a JSON object.

Use the following format:

Question: the input question you must answer
Thought: {{"content": {{"description": "you should always think about what to do", "required": true}}, "tool_to_use_for_user": {{"description": "当调用某个工具缺少必要参数时，填写这个工具的名字", "required": true}}, "known_params": {{"description": "已经提取到的当前要调用工具的参数列表", "required": true}}}} Format the arguments as a JSON object.
Action: the action to take, should be one of [{plugin_name},end_conversation,ASK_USER_FOR_REQUIRED_PARAMS]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)

开始！  

Question: {question}
{reAct_step_str}
"""


FINAL_PROMPT = """Question: {question}
{reAct_step_str}
Final Answer: """
