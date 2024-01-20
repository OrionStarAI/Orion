import os
from typing import Dict

from plugin_task.model import Plugin

PLUGIN_JSON_SCHEMA = [
    {
        "url": os.environ["WEATHER_PLUGIN_URL"],
        "method": "POST",
        "headers": {},
        "request_body": [
            {
                "enum": [],
                "name": "city_name",
                "description": "需要明确城市名称后再调用接口",
                "is_required": True,
                "parament_type": "str",
            }
        ],
        "name_for_human": "查天气",
        "description_for_human": "查询天气的工具,如果发现用户有查询天气的意图会使用该工具",
        "description_for_model": "It can help users check the weather situation, and if the description includes a city, it can be directly queried.",
        "unique_name_for_model": "query_weather",
    },
    {
        "url": "http://gwgp-pt7hznkstln.n.bdcloudapi.com/expTrack?com=auto",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "X-Bce-Signature": os.environ["BCE_APP_CODE"],
        },
        "request_body": [
            {
                "enum": [],
                "name": "nu",
                "description": "需要明确快递单号后再调用接口",
                "is_required": True,
                "parament_type": "str",
            }
        ],
        "name_for_human": "查快递",
        "description_for_human": "查询快递的工具,如果发现用户有查询快递的意图会使用该工具",
        "description_for_model": "It can help users check the express delivery, and if the description includes a express delivery, it can be directly queried.",
        "unique_name_for_model": "query_express",
    },
    {
        "url": "https://api.oioweb.cn/api/common/teladress",
        "method": "GET",
        "headers": {},
        "request_body": [
            {
                "enum": [],
                "name": "mobile",
                "description": "需要明确电话号码后再调用接口",
                "is_required": True,
                "parament_type": "str",
            }
        ],
        "name_for_human": "查电话号码归属地",
        "description_for_human": "查询电话号码归属地的工具,如果发现用户有查询电话号码归属地的意图会使用该工具",
        "description_for_model": "It can help users check the phone number attribution, and if the description includes a phone number, it can be directly queried.",
        "unique_name_for_model": "query_teladress",
    },
    {
        "url": "https://tenapi.cn/v2/bing?format=json",
        "method": "POST",
        "headers": {},
        "request_body": [],
        "name_for_human": "bing每日壁纸",
        "description_for_human": "获取bing每日壁纸的工具,如果发现用户有获取bing每日壁纸会使用该工具",
        "description_for_model": "It can help users Get bing daily wallpapers, and if the description Get bing daily wallpapers, it can be directly queried.",
        "unique_name_for_model": "Get bing daily wallpapers",
    },
]


PLUGINS: Dict[str, Plugin] = {
    plugin["unique_name_for_model"]: Plugin(
        url=plugin["url"],
        method=plugin["method"],
        headers=plugin["headers"],
        request_body=plugin["request_body"],
        name_for_human=plugin["name_for_human"],
        description_for_human=plugin["description_for_human"],
        description_for_model=plugin["description_for_model"],
        unique_name_for_model=plugin["unique_name_for_model"],
    )
    for plugin in PLUGIN_JSON_SCHEMA
}
