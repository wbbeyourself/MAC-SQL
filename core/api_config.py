import os
# todo: set your OPENAI_API_BASE, OPENAI_API_KEY here!
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "your_own_api_base")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_own_api_key")

import openai
openai.api_type = "azure"
openai.api_base = OPENAI_API_BASE
# todo: set your own api_version
openai.api_version = "2023-07-01-preview"
openai.api_key = OPENAI_API_KEY

MODEL_NAME = 'gpt-4-1106-preview' # 128k 版本
# MODEL_NAME = 'gpt-4-32k' # 0613版本
# MODEL_NAME = 'gpt-4' # 0613版本
# MODEL_NAME = 'gpt-35-turbo-16k' # 0613版本