@echo off
chcp 65001

set db_root_path="./data/bird/dev_databases/"
set data_mode="dev"
set diff_json_path="./data/bird/dev.json"
set predicted_sql_json_path="./outputs/bird/predict_dev.json"
set ground_truth_sql_path="./data/bird/dev_gold.sql"
set num_cpus=12
set meta_time_out=30.0
set time_out=60
set mode_gt="gt"
set mode_predict="gpt"

@REM evaluate EX
echo "Evaluate BIRD EX begin!"
python ./evaluation/evaluation_bird_ex.py --db_root_path %db_root_path% ^
    --predicted_sql_json_path %predicted_sql_json_path% ^
    --data_mode %data_mode% ^
    --ground_truth_sql_path %ground_truth_sql_path% ^
    --num_cpus %num_cpus% ^
    --mode_predict %mode_predict% ^
    --diff_json_path %diff_json_path% ^
    --meta_time_out %meta_time_out%
echo "Evaluate EX done!"

@REM evaluate VES
echo "Evaluate BIRD VES begin!"
python ./evaluation/evaluation_bird_ves.py ^
    --db_root_path %db_root_path% ^
    --predicted_sql_json_path %predicted_sql_json_path% ^
    --data_mode %data_mode% ^
    --ground_truth_sql_path %ground_truth_sql_path% ^
    --num_cpus %num_cpus% --meta_time_out %time_out% ^
    --mode_gt %mode_gt% --mode_predict %mode_predict% ^
    --diff_json_path %diff_json_path%
echo "Evaluate VES done!"
