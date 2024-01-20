import json
from typing import Any, Dict, List, Union

import requests
from pydantic import BaseModel


class ReActStep(BaseModel):
    """RaAct 推理步骤"""

    thought: Dict[str, Union[Dict[str, str], str]]
    action: str
    action_input: Dict[str, str]
    observation: Dict[str, Any] = {}

    def to_str(self) -> str:
        s = f"Thought: {self.thought}\n"
        s += f"Action: {self.action}\n"
        s += f"Action Input: {self.action_input}\n"

        if self.observation:
            s += f"Observation: {self.observation}\n"
        return s


class RequstField(BaseModel):
    """请求体字段"""

    enum: List[str]
    name: str
    description: str
    is_required: bool
    parament_type: str

    @property
    def to_simple_dict(self) -> Dict[str, Any]:
        data = {
            "description": self.description,
            "required": self.is_required,
        }
        if self.enum:
            data["enum"] = self.enum
        return data


class Plugin(BaseModel):
    """插件"""

    url: str
    method: str
    headers: Dict[str, str]
    request_body: List[RequstField]
    name_for_human: str
    description_for_human: str
    description_for_model: str
    unique_name_for_model: str

    @property
    def parameter_schema(self) -> str:
        parameter = {}
        for field in self.request_body:
            if not field.is_required:
                continue
            parameter[field.name] = field.to_simple_dict
        return json.dumps(parameter, ensure_ascii=False)

    def run(self, **kwargs):
        """运行插件"""

        response = requests.request(
            self.method.upper(),
            self.url,
            headers=self.headers,
            params=kwargs,
            json=kwargs,
        )
        return response.text

    @property
    def required_parameters(self) -> List[RequstField]:
        """必填参数"""
        return [field for field in self.request_body if field.is_required]
