nohup python attack.py \
    --csv_store_path ./attack_replace.csv \
    --result_store_path ./attack_trans.jsonl \
    --eval_data_file=../../data/test_java2cs.jsonl \
    --sub_data_file=../../data/test_java2cs_substitute.jsonl \
    --eval_batch_size 32 \
    --use_replace \
    --seed 42| tee attack_replace.log

nohup python attack.py \
    --csv_store_path ./attack_insert.csv \
    --result_store_path ./attack_trans.jsonl \
    --eval_data_file=../../data/test_java2cs.jsonl \
    --sub_data_file=../../data/test_java2cs_substitute.jsonl \
    --eval_batch_size 32 \
    --use_insert \
    --seed 42| tee attack_insert.log


nohup python attack.py \
    --csv_store_path ./attack_all.csv \
    --result_store_path ./attack_trans.jsonl \
    --eval_data_file=../../data/test_java2cs.jsonl \
    --sub_data_file=../../data/test_java2cs_substitute.jsonl \
    --eval_batch_size 32 \
    --use_replace \
    --use_insert \
    --seed 42| tee attack_all.log

