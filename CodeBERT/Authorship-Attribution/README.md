# 攻击 CodeBERT，任务：作者署名
* 模型文件huggingface下载链接:mircosoft/codebert-base

## 数据集
数据集在dataset/gcjpy中
运行下列命令，得到train.csv,valid.csv
一个csv文件中包含code,label
```
python process.py
```

## Fine-tune CodeBERT
### On Python dataset

运行下列命令
```shell
cd code
python fine-tune.py
```
或者使用fine-tune.ipynb
* 模型保存在CodeBERTsaved_models中

## Attack

### On Python dataset

#### 得到待替换词汇表
```
cd preprocess
python get_substitutes.py \
    --store_path ./data_folder/processed_gcjpy/valid_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./data_folder/processed_gcjpy/valid.txt \
    --block_size 512
```

#### 攻击

```shell
cd code
source divi_attack.sh
```
运行此命令可进行三种攻击：只进行替换攻击，只进行插入攻击和只进行遗传算法攻击

```shell
cd code
source attack.sh
```
运行此命令进行所有攻击
攻击的log文件中记录了攻击结果

#### 对抗训练
两种对抗训练方式
首先使用adv_preprocess.ipynb文件得到对抗训练的数据集
再使用adversarial_training.ipynb进行两种对抗训练
最后运行
```shell
source adv-training-attack.sh
source adv-fine-tuning-attack.sh
```
即可得到对抗训练后的攻击结果