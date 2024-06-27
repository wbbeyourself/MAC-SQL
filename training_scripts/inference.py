import os
import sys
import h5py
import copy
import json
import argparse
import logging
import re
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, Sequence, List
from tools import *

import torch
import transformers
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
logging.basicConfig(level=logging.DEBUG)  # 修改日志级别为DEBUG

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
DEBUG=False

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForInference(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 4096)
        labels = padding(labels, IGNORE_INDEX, cutoff = 4096)
        if instances[0].get("image") is not None:
            image = torch.cat([instance["image"] for instance in instances], dim=0)
            return dict(
                input_ids=input_ids,
                image=image,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

def padding(inputs, padding_token, cutoff = None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    else:
        cutoff = min(cutoff, max([len(item) for item in inputs]))
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens


def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s) * pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)
    return gathered_s

def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    # print(tokenizer.model_max_length)
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, args):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_jsonl_file(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        if DEBUG:
            list_data_dict = list_data_dict[:1000]
        
        if list_data_dict[0].get("input_ids") is None:  
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)
            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
        else:
            logging.info("Loading tokenized sentences...")
            def truncate(sentence):
                return torch.tensor(sentence[:args.model_max_length] + [tokenizer.eos_token_id] if len(sentence) > args.model_max_length else sentence)
            self.input_ids = [truncate(example["test_input_ids"]) for example in list_data_dict]
            self.labels = [truncate(example["label"]) for example in list_data_dict]
            self.raw = list_data_dict
 

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



@torch.no_grad()
def main(rank, args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    #world_size = torch.cuda.device_count()
    world_size = os.environ["WORLD_SIZE"]
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token='[PAD]'),
            tokenizer=tokenizer,
            model=model,
        )
    
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        }
    )
    tokenizer.truncation_side = 'right'
    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, args=args)
    data_collator = DataCollatorForInference(tokenizer=tokenizer)
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=data_collator, 
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
    )
    generation_config = GenerationConfig(
        temperature=1,
        do_sample=False,
        num_beams=args.beam_size,
        max_new_tokens=args.maxlen_out,
        num_return_sequences=args.return_seq_num,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_ids=tokenizer.eos_token_id
    )
    all_outputs = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        # if step > 5:
        #     break
        generation_output = model.module.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        s = generation_output.sequences
        #s = set_empty_token(s, tokenizer.eos_token_id, tokenizer.pad_token_id)

        bsz = input_ids.shape[0]
        gather_outputs  = sequence_gather(s, world_size, tokenizer.pad_token_id)
        gathered_inputs = sequence_gather(input_ids, world_size, tokenizer.pad_token_id)
       
        gather_outputs  = torch.stack(gather_outputs).reshape(world_size, bsz, args.return_seq_num, -1)
        gathered_inputs = torch.stack(gathered_inputs)
        gather_outputs  = gather_outputs.transpose(0,1).reshape(bsz*world_size * args.return_seq_num, -1)

        gathered_inputs = gathered_inputs.transpose(0,1).reshape(bsz*world_size,-1)
        outputs_string  = tokenizer.batch_decode(gather_outputs , skip_special_tokens=True)
        inputs_string   = tokenizer.batch_decode(gathered_inputs, skip_special_tokens=True)
        # if rank == 0: 
        #     # print(inputs_string[0])
        #     # print(gathered_inputs[0])
        #     # print('+'*10)
        #     # print(gather_outputs[0])
        #     print(outputs_string[0])

        for idx in range(len(inputs_string)):
            temp = []
            for i in range(args.return_seq_num):
                temp.append([inputs_string[idx], outputs_string[args.return_seq_num * idx + i].replace(inputs_string[idx], '')])        
            all_outputs.append(temp)
    
    all_outputs = all_outputs[:len(eval_dataset)]
    os.makedirs(args.out_path, exist_ok=True)
    if rank == 0:
        # assert len(all_outputs) == len(eval_dataset.raw)
        with open(os.path.join(args.out_path, data_path.split('/')[-1]), 'w') as f:
            for idx, (item, raw) in enumerate(zip(all_outputs, eval_dataset.raw)):
                # print('*******************')
                print(item)
                raw['generated'] = item[0][-1]
                f.write(json.dumps(raw) + '\n')
                # f.write(json)
                # print('*******************')
                # f.write(json.dumps(item) + '\n')
    dist.barrier()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--beam_size", type=int, default=4, help="beam size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--use_diverse_beam", type=bool, default=False, help="batch size")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    parser.add_argument("--model_type", default="cmt", type=str, help="config path")
    parser.add_argument("--model_max_length", type=int, default=4096, help="beam size")
    parser.add_argument("--dynamic_load_image", default=True, type=bool, help="config path")
    parser.add_argument("--return_seq_num", default=1, type=int, help="config path")
    parser.add_argument("--maxlen_out", default=512, type=int, help="config path")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)