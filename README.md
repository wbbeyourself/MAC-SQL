## 📖Introduction

This is the official repository for the paper ["MAC-SQL: Multi-Agent Collaboration for text-to-SQL task"](https://arxiv.org/abs/2312.66666).

In this paper, we propose a multi-agent collaborative Text-to-SQL framework MAC-SQL, which comprises three agents: the **Selector**, the **Decomposer**, and the **Refiner**.

<img src="./assets/framework.jpg" align="middle" width="95%">

## ⚡Environment



1. Config your local environment.

```bash
# Use conda to create environment.
conda create -n macsql python=3.8 -y
conda activate macsql
# Install the requirements.
pip install -r requirements.txt
```

2. Config your API_KEY, which used in ```core/api_config.py```.

```bash
set OPENAI_API_BASE = <YOUR_OPENAI_API_BASE>
set OPENAI_API_KEY = <YOUR_OPENAI_API_KEY>
```

## 🔧 Data Preparation



- Download [BIRD dev](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip) and then put in ```<REPO_DIR>/data/bird```

- Download [Spider dev](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ) and then put in ```<REPO_DIR>/data/spider```

- The expected repo structure should be like as:
    ```bash
    MAC-SQL
    ├─data
      ├─bird
      │  │─dev.json
      │  │─dev.sql
      │  │─dev_gold.sql
      │  │─dev_tables.json
      │  └─dev_databases
      │      ├─california_schools
      │      │  │─.DS_Store
      │      │  │─california_schools.sqlite
      │      │  │
      │      │  └─database_description
      │      │     │─frpm.csv
      │      │     │─satscores.csv
      │      │     └─schools.csv
      |      ├─... 
      └─spider
          │─dev.json
          │─dev_gold.sql
          │─README.txt
          │─tables.json
          └─database
              ├─academic
              │    │─academic.sqlite
              │    └─schema.sql
              │
              ├─...
    
    ```

## 🚀 Run


We support different platforms in relative scripts. 

You should open code comments for different usage.

- On `Linux/Mac OS`
```bash
# run bird
sh run_bird.sh
# run spider
sh run_spider.sh
```

- On `Windows OS`
```bash
# run bird
run_bird.bat
# run spider
run_spider.bat
```

## 📝Evaluation Dataset

We evaluate our method on both BIRD dataset and Spider dataset.

EX: Execution Accuracy(%)

VES: Valid Efficiency Score(%)

Refer to our paper for the details.


## 💬Citation



If you find our work is helpful, please cite as:

```text
@article{wang2023macsql,
  title={MAC-SQL: Multi-Agent Collaboration for text-to-SQL task},
  author={Wang, Bing etc.},
  journal={arXiv preprint arXiv:2312.66666},
  year={2023}
}

```

## 👍Contributing


We welcome contributions and suggestions!
