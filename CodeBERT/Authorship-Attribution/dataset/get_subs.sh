python get_substitutes.py \
    --store_path ./data_folder/processed_gcjpy/train_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./data_folder/processed_gcjpy/train.txt \
    --block_size 512