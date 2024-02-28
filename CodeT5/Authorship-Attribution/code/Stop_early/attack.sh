
python attack.py \
    --csv_store_path ./attack_2.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --accpatance 2 \
    --use_replace \
    --seed 42| tee attack_2.log

python attack.py \
    --csv_store_path ./attack_4.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --accpatance 4 \
    --use_replace \
    --seed 42| tee attack_4.log

python attack.py \
    --csv_store_path ./attack_8.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --accpatance 8 \
    --use_replace \
    --seed 42| tee attack_8.log

python attack.py \
    --csv_store_path ./attack_sort2.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --accpatance 2 \
    --use_replace \
    --sort_substitute \
    --seed 42| tee attack_sort2.log


python attack.py \
    --csv_store_path ./attack_sort4.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --accpatance 4 \
    --use_replace \
    --sort_substitute \
    --seed 42| tee attack_sort4.log


python attack.py \
    --csv_store_path ./attack_sort8.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --eval_batch_size 32 \
    --accpatance 8 \
    --use_replace \
    --sort_substitute \
    --seed 42| tee attack_sort8.log


