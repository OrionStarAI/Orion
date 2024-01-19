 **# Gradio演示**

## 快速入门

1. 安装依赖项

```bash
pip install -r requirements.txt
```

2. 设置模型推理服务器

3. 设置环境变量

```bash
export BACKEND_HOST="***"  # 模型推理服务器的主机
export MODEL_NAME="***"  # 模型的名称
export API_KEY=="***"  # 模型的 API 密钥，如果没有身份验证可以忽略
export CHAT_ENDPOINT="***"  # 聊天使用的模型的接口，用户可以使用不同的模型来完成不同的任务
export DOC_QA_ENDPOINT="***"  # 文档问答使用的模型的接口，用户可以使用不同的模型来完成不同的任务
export PLUGIN_ENDPOINT="***"  # 插件使用的模型的接口，用户可以使用不同的模型来完成不同的任务
export QA_GENERATOR_ENDPOINT="***"  # 问答生成器使用的模型的接口，用户可以使用不同的模型来完成不同的任务
export EXTRACT_ENDPOINT="***"  # 信息提取使用的模型的接口，用户可以使用不同的模型来完成不同的任务

# 插件可以根据实际情况进行替换
export WEATHER_PLUGIN_URL="***"  # 天气插件的URL
export BCE_APP_CODE="***"  # bce的app code
```

4. 运行演示

```bash
gradio app.py
```
