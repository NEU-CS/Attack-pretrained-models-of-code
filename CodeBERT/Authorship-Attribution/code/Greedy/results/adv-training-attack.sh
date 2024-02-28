python attack.py \
    --csv_store_path ./attack_gi.csv \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.csv \
    --model_name_or_path=CODEBERT-ADV-FINE-TUNING \
    --tokenizer_name=CODEBERT-ADV-FINE-TUNING \
    --eval_batch_size 32 \
    --use_ga\
    --seed 42| tee attack_gcjpy.log

mv attack_gcjpy.log attack_adv-training.log
mv attack_gi.csv attack_adv-training.csv