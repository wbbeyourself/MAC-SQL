import jsonlines
import os
import h5py
from PIL import Image
import numpy as np
import transformers
import torch
import math
import copy
import multiprocessing as mp
import tqdm
import traceback
from time import sleep
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_MAX_LENGTH = 5000

BASE_MODEL_DIR="/your/path/to/llms_root_dir/CodeLlama-7b-hf"
DATA_DIR = './data'

tokenizer = transformers.AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    cache_dir=None,
    model_max_length=MODEL_MAX_LENGTH,
    padding_side="right",
    use_fast=False,
)
special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)


class MPLogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def error(msg, *args):
        return mp.get_logger().error(msg, *args) 

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result


def write_json_file(path, objs):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with jsonlines.open(path,'w') as w:
        for obj in objs:
            w.write(obj)
    print(f"Successfully saving to {path}: {len(objs)} samples")


def _tokenize_fn(text, tokenizer):
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = labels = tokenized.input_ids[0]
    input_ids_lens = labels_lens = tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def tokenize_text(obj):
    obj["source"] = obj["source"]
    obj["target"] = f"{obj['target']}{tokenizer.eos_token}"
    tokenized_example = _tokenize_fn(obj["source"] + obj["target"], tokenizer)
    tokenized_source = _tokenize_fn(obj["source"], tokenizer)
    input_ids = tokenized_example["input_ids"]
    source_len = tokenized_source["input_ids_lens"]
    label = copy.deepcopy(input_ids)
    label[:source_len] = IGNORE_INDEX
    obj["test_input_ids"] = tokenized_source["input_ids"].tolist() # # input ids
    obj["input_ids"] = input_ids.tolist() # input + output + EOS ids
    obj["label"] = label.tolist() # len=len(input + output + EOS), ignore input id part
    return obj

def read_jsonl_file(path, max_sentence=None):
    data = []
    with jsonlines.open(path, "r") as r:
        for i, obj in tqdm.tqdm(enumerate(r)):
            if max_sentence is not None and i >= max_sentence:
                return data
            data.append(obj)
    print(f"Successfully loading {len(data)} examples from {path}")
    return data

def get_llama2_prompt(dialog):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{system}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{system}\n\n### Response:"
        ),
        "system": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{system}\n\n"
        ),
        "history": (
            "### Input:\n{input}\n\n### Response:\n{response}\n\n"
        ),
        "query": (
            "### Input:\n{input}\n\n### Response:\n"
        ),
        "response": (
            "{response}"
        )
    }
    prompt = ""
    if dialog[0]["role"] == "system":
        sys_text = dialog[0]["content"].strip()
        if sys_text != '':
            prompt = PROMPT_DICT["system"].format_map({"system": dialog[0]["content"]})
        dialog = dialog[1:]
    for i in range(0, len(dialog) - 2, 2):
        prompt += PROMPT_DICT["history"].format_map({
            "input": dialog[i]["content"], 
            "response": dialog[i + 1]["content"]
        })
    if len(dialog) == 1:
        assert dialog[-1]["role"] == "user"
        prompt += PROMPT_DICT["query"].format_map({"input": dialog[-1]["content"]})
        response = ""
    else:
        assert dialog[-2]["role"] == "user" and dialog[-1]["role"] == "assistant"
        prompt += PROMPT_DICT["query"].format_map({"input": dialog[-2]["content"]})
        response = PROMPT_DICT["response"].format_map({"response": dialog[-1]["content"]})    
    return prompt, response

def build_sft_data(worker_id = 0, objs = None):
    output_objs = []
    for obj in tqdm.tqdm(objs, position = worker_id, desc=f"worker_id: {worker_id}"):
        obj["source"], obj["target"] = get_llama2_prompt(obj["messages"])
        obj = tokenize_text(obj)
        # 超长直接丢弃了
        length = len(obj["input_ids"])
        if length >= tokenizer.model_max_length:
            print(f'long length: {length}')
            continue
        else:
            print(f'normal length: {length}')
        output_objs.append(obj)
    return output_objs

def construct_and_merge_data(
        worker = 16,
        split="train",
        chunk_size = 1000,
        input_data_path = "",
        output_data_dir = f"{DATA_DIR}/processed"
    ):
    output_objs = []
    results = []

    os.makedirs(output_data_dir, exist_ok=True)
    
    if not os.path.isfile(input_data_path):
        file_names = [file_name for file_name in os.listdir(input_data_path) if split in file_name]
    output_objs = []
    p = mp.Pool(worker)
    if worker == 1:
        for file_name in file_names:
            objs = read_jsonl_file(f"{input_data_path}/{file_name}")
            output_objs.extend(build_sft_data(worker_id=0, objs=objs))
    else:
        if os.path.isfile(input_data_path):
            objs = read_jsonl_file(input_data_path)
            print('len(objs):', len(objs))
            chunk_num = math.ceil(len(objs) / float(chunk_size))
            worker_id = 0
            print(f"chunk_num: {chunk_num}")
            sleep(3)
            for i in range(0, chunk_num):
                results.append(p.apply_async(MPLogExceptions(build_sft_data), args=(worker_id, objs[i * chunk_size: (i + 1) * chunk_size])))
                worker_id += 1
            p.close()
            p.join()
            for result in results:
                output_objs.extend(result.get())
            print('len(output_objs): ', len(output_objs))
            write_json_file(f"{output_data_dir}/{os.path.basename(input_data_path)}", output_objs)
        else:
            worker_id = 0
            for file_name in file_names:
                dataset_name = file_name.split("_")[-2]
                objs = read_jsonl_file(f"{input_data_path}/{file_name}")
                chunk_num = math.ceil(len(objs) / float(chunk_size))
                for i in range(0, chunk_num):
                    results.append(p.apply_async(MPLogExceptions(build_sft_data), args=(worker_id, objs[i * chunk_size: (i + 1) * chunk_size], dataset_name)))
                    worker_id += 1
            p.close()
            p.join()
            for result in results:
                output_objs.extend(result.get())
            write_json_file(f"{output_data_dir}/evol_code_alpaca_v1.jsonl", output_objs)


if __name__ == "__main__":
    DATA_DIR = './data'
    construct_and_merge_data(
        worker = 32,
        input_data_path=f"{DATA_DIR}/raw/sql-llama-instruct-v0.5.jsonl",
        output_data_dir=f"{DATA_DIR}/processed"
    )