# SQL-Llama


## fastchat deployment

```bash
conda create -n fastchat python=3.10.0 -y
# my fschat version 0.2.34
conda activate fastchat
pip3 install "fschat[model_worker,webui]"
pip3 install openai==0.28.1
```

It is recommended to use tmux in Linux environment and start Controller, Model Worker, and API Server in separate windows respectively.

### Run Controller

```bash
python3 -m fastchat.serve.controller --port 21000  --host 0.0.0.0
```


### Run Model Worker

```bash
CUDA_VISIBLE_DEVICES=1  python3 -m fastchat.serve.model_worker --model-name CodeLlama-7b-hf  --model-path  /your/path/to/SQL-Llama-v0.5  --worker-address http://0.0.0.0:30002 --port 30002 --host 0.0.0.0 --controller-address http://0.0.0.0:21000
```

Once "Uvicorn running on http://0.0.0.0:30002 (Press CTRL+C to quit)" appears, it means it is okay.

Multiple workers can be started, as shown below. You only need to modify the port.

```bash
CUDA_VISIBLE_DEVICES=2  python3 -m fastchat.serve.model_worker --model-name CodeLlama-7b-hf  --model-path  /your/path/to/SQL-Llama-v0.5  --worker-address http://0.0.0.0:30003 --port 30003 --host 0.0.0.0 --controller-address http://0.0.0.0:21000

CUDA_VISIBLE_DEVICES=3  python3 -m fastchat.serve.model_worker --model-name CodeLlama-7b-hf  --model-path  /your/path/to/SQL-Llama-v0.5  --worker-address http://0.0.0.0:30004 --port 30004 --host 0.0.0.0 --controller-address http://0.0.0.0:21000
```


### Run API Server
```bash
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0  --port 8000 --controller-address http://0.0.0.0:21000
```

Once "Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)" appears, it means it is okay.


### try demo
```bash
cd scripts
python fastchat_demo.py
```

If you see the json output, it means the api server is running and it is okay.

```json
{
    "id": "chatcmpl-3Ad22upokgv2ggGThKAe6s",
    "object": "chat.completion",
    "created": 1710840181,
    "model": "CodeLlama-7b-hf",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here's the quick sort in Python:\n```python\ndef quick_sort(array):\n    if len(array) <= 1:\n        return array\n    else:\n        pivot = array[0]\n        lesser = [i for i in array[1:] if i <= pivot]\n        greater = [i for i in array[1:] if i > pivot]\n        return quick_sort(lesser) + [pivot] + quick_sort(greater)\n```\n"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 583,
        "total_tokens": 693,
        "completion_tokens": 110
    }
}
```



