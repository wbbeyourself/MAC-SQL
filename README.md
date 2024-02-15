## 📖Introduction

This is the official repository for the paper "MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL".

In this paper, we propose a multi-agent collaborative Text-to-SQL framework MAC-SQL, which comprises three agents: the **Selector**, the **Decomposer**, and the **Refiner**.

<img src="./assets/framework.jpg" align="middle" width="95%">

## ⚡Environment

1. Config your local environment.

```bash
conda create -n macsql python=3.9 -y
conda activate macsql
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

Note: we use `openai==0.28.1`, which use `openai.ChatCompletion.create` to call api.

2. Edit openai config at **core/api_config.py**, and set related environment variables of Azure OpenAI API.

Currently, we use `gpt-4-1106-preview` (128k version) by default, which is 2.5 times less expensive than the `gpt-4 (8k)` on average.

```bash
export OPENAI_API_BASE="YOUR_OPENAI_API_BASE"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## 🔧 Data Preparation

In order to prepare the data more quickly, I have packaged the files including the databases of the BIRD dataset and the Spider dataset into `data.zip` and uploaded them. 
All files were downloaded on December 19, 2023, ensuring they are the latest version at that moment. 
The download links are available on Baidu Disk and Google Drive.

After downloading the `data.zip` file, you should delete the existing data folder in the project directory and replace it with the unzipped data folder from `data.zip`.


## 🚀 Run

The run script will first run 5 examples in Spider to check environment.
You should open code comments for different usage.

- `run.sh` for Linux/Mac OS
- `run.bat` for Windows OS

For SQL execution demo, you can use `app_bird.py` or `app_spider.py` to get the execution result of your SQL query.

```bash
cd ./scripts
python app_bird.py
python app_spider.py
```

## 📝Evaluation Dataset

We evaluate our method on both BIRD dataset and Spider dataset.

EX: Execution Accuracy(%)

VES: Valid Efficiency Score(%)

Refer to our paper for the details.


## 🌟 Project Structure

```txt
├─data # store datasets and databases
|  ├─spider
|  ├─bird
├─core
|  ├─agents.py       # define three agents class
|  ├─api_config.py   # OpenAI API ENV config
|  ├─chat_manager.py # manage the communication between agents
|  ├─const.py        # prompt templates and CONST values
|  ├─llm.py          # api call function and log print
|  ├─utils.py        # utils function
├─scripts            # sqlite execution flask demo
|  ├─app_bird.py
|  ├─app_spider.py
|  ├─templates
├─evaluation # evaluation scripts
|  ├─evaluation_bird_ex.py
|  ├─evaluation_bird_ves.py
|  ├─evaluation_spider.py
├─bad_cases
|  ├─badcase_BIRD(dev)_examples.xlsx
|  └badcase_Spider(dev)_examples.xlsx
├─evaluation_bird_ex_ves.sh # bird evaluation script
├─README.md
├─requirements.txt
├─run.py # main run script
├─run.sh # generation and evaluation script
```
