# SQL-Llama-Fintuning

## Introduction

SQL-Llama is finetuned base on CodeLlama-7b-hf.
You should config your llm_root_dir in `finetuning.sh`.

## Requirements
See requirements.txt

## Data
Download the `sql-llama-data.zip` from [Baidu Dsik](https://pan.baidu.com/s/1yaEBsSN894O7MlBrckciKw?pwd=htwt) or [Google Drive](https://drive.google.com/file/d/1_3s88Op1PCZo50RsHcx5m2Bj_n05PPn4/view?usp=sharing).
Unzip `sql-llama-data.zip` and get the data dir, which contains sql-llama-instruct-v0.5.jsonl (3375 instances).


## Finetuning Details

- Computation Resource Requirements: A100(40G) * 8
- Training Time: x hours