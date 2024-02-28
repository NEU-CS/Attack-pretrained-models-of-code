python attack.py \
    --csv_store_path ./attack_beamsize8.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --beam_size 8\
    --eval_batch_size 32 \
    --use_replace \
    --seed 42| tee attack_beamsize8.log 