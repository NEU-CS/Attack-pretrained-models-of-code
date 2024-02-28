python attack.py \
    --csv_store_path ./attack_replace.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --use_replace \
    --seed 42| tee attack_replace.log

python attack.py \
    --csv_store_path ./attack_insert.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --use_insert \
    --seed 42| tee attack_insert.log


python attack.py \
    --csv_store_path ./attack_all.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --use_replace \
    --use_insert \
    --seed 42| tee attack_all.log

