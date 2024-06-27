# -*- coding: utf-8 -*-
from tools import *
from core.chat_manager import ChatManager
from core.const import SYSTEM_NAME
from tqdm import tqdm
import time
import argparse
import sys
import os
import json
import traceback


def init_spider_message(idx: int, item: dict) -> dict:
    """
    Construct message for text-to-SQL task
    :param idx: start from 0
    :param item: one sample of dataset
    :return: initial message object of group chat
    """
    db_id, query, evidence, gt = item['db_id'], item['question'], str(""), item['query']
    difficulty = eval_hardness(item['sql'])
    user_message = {
        "idx": idx,
        "db_id": db_id,
        "query": query,
        "evidence": evidence,
        "extracted_schema": {},
        "ground_truth": gt,
        "difficulty": difficulty,
        "send_to": SYSTEM_NAME
    }
    return user_message


def init_bird_message(idx: int, item: dict, db_path: str=None, use_gold_schema: bool = False) -> dict:
    """
    Construct message for text-to-SQL task
    :param idx: start from 0
    :param item: one sample of dataset
    :return: initial message object of group chat
    """
    db_id, query, evidence, gt, difficulty = item['db_id'], \
                                             item['question'], \
                                             item['evidence'], \
                                             item.get('SQL', ''), \
                                             item.get('difficulty', 'simple')
    
    gold_schema_path = './data/bird/dev_gold_schema.json'
    gold_schema = {}
    all_gold_schema_dict = {}
    key = f"{db_id.strip()}\t{query.strip()}"
    if use_gold_schema:
        if os.path.exists(gold_schema_path):
            all_gold_schema_dict = load_json_file(gold_schema_path)
        if key in all_gold_schema_dict:
            gold_schema = all_gold_schema_dict[key]
        else:
            raise ValueError(f"Can't find gold schema for {key}")
    
    user_message = {
        "idx": idx,
        "db_id": db_id,
        "query": query,
        "evidence": evidence,
        "extracted_schema": gold_schema if gold_schema else {},
        "ground_truth": gt,
        "difficulty": difficulty,
        "send_to": SYSTEM_NAME
    }
    return user_message


def run_batch(dataset_name, input_file, output_file, db_path, tables_json_path, start_pos=0, log_file=None, dataset_mode='dev', use_gold_schema=False, without_selector=False):
    chat_manager = ChatManager(data_path=db_path,
                               tables_json_path=tables_json_path,
                               log_path=log_file,
                               dataset_name=dataset_name,
                               model_name='gpt-4',
                               lazy=True,
                               without_selector=without_selector)
    # load dataset
    batch = load_json_file(input_file)
    # resume from last checkpoint
    finished_ids = set()
    if os.path.exists(output_file):
        output_data_lst = load_jsonl_file(output_file)
        for o in output_data_lst:
            finished_ids.add(o['idx'])
    unfinished_ids = [n for n in range(len(batch)) if n not in finished_ids and n >= start_pos]
    print(f"len(unfinished_data) = {len(unfinished_ids)}")

    # add question_id if needed
    for k, item in enumerate(batch):
        if 'question_id' not in item:
            item['question_id'] = k

    # skip some json data
    excluded_db_ids = []
    if dataset_mode == 'train':
        exclude_txt = './data/bird_train/excluded_db_ids.txt'
        excluded_db_ids = read_txt_file(exclude_txt)
    new_batch = []
    exclude_db_json_cnt = 0 # for exclude some dbs in bird train set
    for k, item in enumerate(batch):
        q_id = item['question_id']
        if q_id not in unfinished_ids:
            continue
        if dataset_mode == 'train':
            # skip excluded db_id
            if item['db_id'] in excluded_db_ids:
                exclude_db_json_cnt += 1
                continue
        new_batch.append(item)
    
    if exclude_db_json_cnt:
        print(f"excluded {exclude_db_json_cnt} excluded db json data")
    time.sleep(2)
    batch = new_batch


    # generate SQL one by one, and save result one by one
    with open(output_file, 'a+', encoding='utf-8') as fp:
        total_num = len(batch)
        for cur_idx, item in tqdm(enumerate(batch), total=total_num):
            idx = item['question_id']
            db_id = item['db_id']
            print(f"\n\nprocessing: {cur_idx}/{total_num}\n\n", flush=True)
            if idx not in unfinished_ids: continue
            if dataset_name == "spider":
                user_message = init_spider_message(idx, item)  # imitate user send a question to system
            elif dataset_name == "bird":
                user_message = init_bird_message(idx, item, db_path=db_path, use_gold_schema=use_gold_schema)  # imitate user send a question to system
            try:
                chat_manager.start(user_message)
                try:
                    del user_message['desc_str']
                    del user_message['fk_str']
                    del user_message['send_to']
                except:
                    pass
                print(json.dumps(user_message, ensure_ascii=False), file=fp, flush=True)
            except Exception as e:
                # for debug
                traceback.print_exc()
                print(f"Exception: {e}, sleep 20 seconds.", flush=True)
                time.sleep(20)
                # raise Exception(str(e))
            print(f"\n\ndeal {cur_idx+1}/{total_num} done!\n\n")
        print(f"Result dump into {output_file}", file=sys.stdout, flush=True)

    # export evaluation results
    out_dir = os.path.dirname(output_file)
    
    # transfer SQL result to supportable BIRD format
    if dataset_name == "bird":
        evaluation_file_path = f"{out_dir}/predict_{dataset_mode}.json"
        with open(evaluation_file_path, 'w', encoding='utf8') as fout:
            output_json_list = load_jsonl_file(output_file)
            output_json_list = sorted(output_json_list, key=lambda i: i['idx'])
            eval_tuple_lst = []
            for o in output_json_list:
                pred_sql = o['pred'].strip()
                pred_sql = replace_multiple_spaces(pred_sql)
                sql_and_db_str = pred_sql + '\t' + '----- bird -----' + '\t' + o['db_id']
                obj = [o['query'], sql_and_db_str]
                eval_tuple_lst.append(obj)
            json.dump(eval_tuple_lst, fp=fout, ensure_ascii=False, indent=2)
            print(f"BIRD format file dump into {evaluation_file_path}")
    elif dataset_name == "spider":
        evaluation_file_path = f"{out_dir}/pred_{dataset_mode}.sql"
        spider_sql_lst = []
        output_json_lst = load_jsonl_file(output_file)
        for output_json in output_json_lst:
            pred_sql = output_json['pred']
            pred_sql = replace_multiple_spaces(pred_sql)
            spider_sql_lst.append(pred_sql.strip() + '\n')
        save_file(evaluation_file_path, spider_sql_lst)
        print(f"Spider format file dump into {evaluation_file_path}")
    else:
        raise NotImplementedError


def check_all_paths(args):
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} not found")
    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"Database path {args.db_path} not found")
    if not os.path.exists(args.tables_json_path):
        raise FileNotFoundError(f"Tables json path {args.tables_json_path} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='spider', choices=['spider', 'bird'], help='dataset name')
    parser.add_argument('--dataset_mode', type=str, default='dev', choices=['train', 'dev', 'test'], help='dataset mode')
    parser.add_argument('--input_file', type=str, required=True, help='path to dataset input')
    parser.add_argument('--db_path', type=str, required=True, help='path to databases in dataset')
    parser.add_argument('--tables_json_path', type=str, default=None, help='path to tables.json')
    parser.add_argument('--output_file', type=str, required=True, help='path to predicted output')
    parser.add_argument('--log_file', type=str, default='', help='path to log file if needed')
    parser.add_argument('--start_pos', type=int, default=0, help='start position of a batch')
    parser.add_argument('--use_gold_schema', action='store_true', default=False)
    parser.add_argument('--without_selector', action='store_true', default=False)
    args = parser.parse_args()
    # 打印args中的键值对
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    check_all_paths(args)

    # pretty print args json
    args_json_str = json.dumps(vars(args), indent=2, ensure_ascii=False)
    print(f"args:\n{args_json_str}")
    time.sleep(3)

    run_batch(
        dataset_name=args.dataset_name,
        dataset_mode=args.dataset_mode,
        input_file=args.input_file,
        output_file=args.output_file,
        db_path=args.db_path,
        tables_json_path=args.tables_json_path,
        log_file=args.log_file,
        start_pos=args.start_pos,
        use_gold_schema=args.use_gold_schema,
        without_selector=args.without_selector
    )
