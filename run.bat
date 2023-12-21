@echo off
chcp 65001

@REM default using gpt-4-32k

@REM Generate SQL on foo dataset for env test
@REM This will get ./outputs/foo/output_bird.json and ./outputs/foo/predict_test.json
python ./run.py --dataset_name "bird" ^
   --dataset_mode="test" ^
   --input_file "./data/foo/test.json" ^
   --db_path "./data/foo/test_databases" ^
   --tables_json_path "./data/foo/test_tables.json" ^
   --output_file "./outputs/foo/output_bird.json" ^
   --log_file "./outputs/foo/log.txt"

echo "Generate SQL on env test data done!"


@REM #################### BIRD dev 【run】count=1534 #########
@REM Generate SQL on BIRD dev dataset
@REM python ./run.py --dataset_name="bird" ^
@REM    --dataset_mode="dev" ^
@REM    --input_file="./data/bird/dev.json" ^
@REM    --db_path="./data/bird/dev_databases/" ^
@REM    --tables_json_path "./data/bird/dev_tables.json" ^
@REM    --output_file="./outputs/bird/output_bird.json" ^
@REM    --log_file="./outputs/bird/log.txt"


@REM #################### BIRD dev 【evaluation】=1534, see evaluation_bird_ex_ves.bat #########


@REM #################### Spider dev 【run】count=1034 #########
@REM Generate SQL on BIRD dev dataset
@REM python ./run.py --dataset_name "spider" ^
@REM    --dataset_mode="dev" ^
@REM    --input_file "./data/spider/dev.json" ^
@REM    --db_path "./data/spider/database" ^
@REM    --tables_json_path "./data/spider/tables.json" ^
@REM    --output_file "./outputs/spider/output_spider.json" ^
@REM    --log_file "./outputs/spider/log.txt"

@REM #################### Spider dev 【evaluation】EX and EM count=1034 #########
@REM python ./evaluation/evaluation_spider.py ^
@REM    --gold "./data/spider/dev_gold.sql" ^
@REM    --db "./data/spider/database" ^
@REM    --table "./data/spider/tables.json" ^
@REM    --pred "./outputs/spider/pred_dev.sql" ^
@REM    --etype "all" ^
@REM    --plug_value ^
@REM    --keep_distinct ^
@REM    --progress_bar_for_each_datapoint


echo "Done!"