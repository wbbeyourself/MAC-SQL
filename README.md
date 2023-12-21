## 📖Introduction

This is the official repository for the paper ["MAC-SQL: Multi-Agent Collaboration for text-to-SQL task"](https://arxiv.org/abs/2312.11242).

In this paper, we propose a multi-agent collaborative Text-to-SQL framework MAC-SQL, which comprises three agents: the **Selector**, the **Decomposer**, and the **Refiner**.

<img src="./assets/framework.jpg" align="middle" width="95%">


# 🔥 Updates
- [**2023-12-26**] We have updated the paper, with updates mainly focusing on the title, abstract, introduction, some details, and appendix. In addition, we give some bad case examples on `bad_cases` folder, check it out!
- [**2023-12-19**] We released our first version [paper](https://arxiv.org/abs/2312.11242), [code](https://github.com/wbbeyourself/MAC-SQL). Check it out!



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

Currently, we use `gpt-4` (8k version) by default.

```bash
export OPENAI_API_BASE="YOUR_OPENAI_API_BASE"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## 🔧 Data Preparation

In order to prepare the data more quickly, I have packaged the files including the databases of the BIRD dataset and the Spider dataset into `data.zip` and uploaded them. 
All files were downloaded on December 19, 2023, ensuring they are the latest version at that moment. 
The download links are available on [Baidu Disk](https://pan.baidu.com/s/1LkOtJEjNCG-N-ozq4mbuag?pwd=6ak5) and [Google Drive](https://drive.google.com/file/d/1dXHsYDziA8NwZuyjDaFV3j9CPjYdNuyP/view?usp=drive_link).

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


## 💬Citation


If you find our work is helpful, please cite as:

```text
@misc{wang2023macsql,
      title={MAC-SQL: Multi-Agent Collaboration for Text-to-SQL}, 
      author={Bing Wang and Changyu Ren and Jian Yang and Xinnian Liang and Jiaqi Bai and Qian-Wen Zhang and Zhao Yan and Zhoujun Li},
      year={2023},
      eprint={2312.11242},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 👍Contributing


We welcome contributions and suggestions!
