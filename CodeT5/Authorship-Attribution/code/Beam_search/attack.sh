python attack.py \
    --csv_store_path ./attack_beamsize1.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --beam_size 1\
    --eval_batch_size 32 \
    --use_replace \
    --seed 42| tee attack_beamsize1.log


python attack.py \
    --csv_store_path ./attack_beamsize2.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --beam_size 2\
    --eval_batch_size 32 \
    --use_replace \
    --seed 42| tee attack_beamsize2.log 


python attack.py \
    --csv_store_path ./attack_beamsize4.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --beam_size 4\
    --eval_batch_size 32 \
    --use_replace \
    --seed 42| tee attack_beamsize4.log 



python attack.py \
    --csv_store_path ./attack_beamsize1_sort.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --beam_size 1\
    --eval_batch_size 32 \
    --use_replace \
    --sort_substitute \
    --seed 42| tee attack_beamsize1_sort.log 


python attack.py \
    --csv_store_path ./attack_beamsize2_sort.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --beam_size 2\
    --eval_batch_size 32 \
    --use_replace \
    --sort_substitute \
    --seed 42| tee attack_beamsize2_sort.log 


python attack.py \
    --csv_store_path ./attack_beamsize4_sort.csv \
    --eval_data_file=../../dataset/data_folder/processed_gcjpy/valid.csv \
    --beam_size 4\
    --eval_batch_size 32 \
    --use_replace \
    --sort_substitute \
    --seed 42| tee attack_beamsize4_sort.log 

