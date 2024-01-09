import sys
import json
import time
from core.api_config import *

MAX_TRY = 5

# 用来传递外面的字典进来
world_dict = {}

log_path = None
api_trace_json_path = None
total_prompt_tokens = 0
total_response_tokens = 0


def init_log_path(my_log_path):
    global total_prompt_tokens
    global total_response_tokens
    global log_path
    global api_trace_json_path
    log_path = my_log_path
    total_prompt_tokens = 0
    total_response_tokens = 0
    dir_name = os.path.dirname(log_path)
    os.makedirs(dir_name, exist_ok=True)

    # 另外一个记录api调用的文件
    api_trace_json_path = os.path.join(dir_name, 'api_trace.json')


def api_func(prompt:str):
    global MODEL_NAME
    print(f"\nUse OpenAI model: {MODEL_NAME}\n")
    if MODEL_NAME.startswith('CodeLlama'):
        openai.api_base = 'http://0.0.0.0:8000/v1'
    response = openai.ChatCompletion.create(
        engine=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    text = response['choices'][0]['message']['content'].strip()
    prompt_token = response['usage']['prompt_tokens']
    response_token = response['usage']['completion_tokens']
    return text, prompt_token, response_token


def safe_call_llm(input_prompt, **kwargs) -> str:
    """
    函数功能描述：输入 input_prompt ，返回 模型生成的内容（内部自动错误重试5次，5次错误抛异常）
    """
    global MODEL_NAME
    global log_path
    global api_trace_json_path
    global total_prompt_tokens
    global total_response_tokens
    global world_dict

    for i in range(5):
        try:
            if log_path is None:
                # print(input_prompt)
                sys_response, prompt_token, response_token = api_func(input_prompt)
                print(f"\nsys_response: \n{sys_response}")
                print(f'\n prompt_token,response_token: {prompt_token} {response_token}\n')
            else:
                # check log_path and api_trace_json_path is not None
                if (log_path is None) or (api_trace_json_path is None):
                    raise FileExistsError('log_path or api_trace_json_path is None, init_log_path first!')
                with open(log_path, 'a+', encoding='utf8') as log_fp, open(api_trace_json_path, 'a+', encoding='utf8') as trace_json_fp:
                    print('\n' + f'*'*20 +'\n', file=log_fp)
                    print(input_prompt, file=log_fp)
                    print('\n' + f'='*20 +'\n', file=log_fp)
                    sys_response, prompt_token, response_token = api_func(input_prompt)
                    print(sys_response, file=log_fp)
                    print(f'\n prompt_token,response_token: {prompt_token} {response_token}\n', file=log_fp)

                    if len(world_dict) > 0:
                        world_dict = {}
                    
                    if len(kwargs) > 0:
                        world_dict = {}
                        for k, v in kwargs.items():
                            world_dict[k] = v
                    # prompt response to world_dict
                    world_dict['response'] = '\n' + sys_response.strip() + '\n'
                    world_dict['input_prompt'] = input_prompt.strip() + '\n'

                    world_dict['prompt_token'] = prompt_token
                    world_dict['response_token'] = response_token
                    

                    total_prompt_tokens += prompt_token
                    total_response_tokens += response_token

                    world_dict['cur_total_prompt_tokens'] = total_prompt_tokens
                    world_dict['cur_total_response_tokens'] = total_response_tokens

                    # world_dict to json str
                    world_json_str = json.dumps(world_dict, ensure_ascii=False)
                    print(world_json_str, file=trace_json_fp)

                    world_dict = {}
                    world_json_str = ''

                    print(f'\n total_prompt_tokens,total_response_tokens: {total_prompt_tokens} {total_response_tokens}\n', file=log_fp)
                    print(f'\n total_prompt_tokens,total_response_tokens: {total_prompt_tokens} {total_response_tokens}\n')
            return sys_response
        except Exception as ex:
            print(ex)
            print(f'Request {MODEL_NAME} failed. try {i} times. Sleep 20 secs.')
            time.sleep(20)

    raise ValueError('safe_call_llm error!')


if __name__ == "__main__":
    res = safe_call_llm('我爸妈结婚为什么不邀请我？')
    print(res)
