# 针对于代码预训练模型的黑盒攻击技术
问题建模为m^n数量级个path上，选择一个最好的path；
本项目基于贪心解码的思想，扩展了beam search
## 代码变换方式
本项目目前使用rename identifiers,dead code insert两种；

## 模型
本项目目前对CodeBERT，CodeT5，CodeLLaMA以及DeepSeek四种代码预训练模型进行攻击；
其中前两种为小模型，后两种为大模型。且这四种模型有encoder-only的bert式模型，有encoder-decoder模型，也有decoder-only模型

## 任务
目前有一个分类任务：Authorship-Attribution作者署名任务：给定一段Python代码，预测完成这段代码的作者是谁
有一个生成任务：Code Translation：将java代码翻译为c-sharp代码

## 文件组织结构  

```
project
│   README.md
│   utils.py(utils of mine)
|   utilsALERT.py(utils of baseline) 
│
└───Model(e.g. DeepSeek)
    │   
    │   
    │
    └───Task(e.g. Authorship-Attribution)
    |    │   
    |    └───code
    |    │
    |    └───dataset
    |
    └─── python_parser(code parser of program languages)
```

code目录中包含模型fine-tune,attack的代码；注意，模型在攻击前需要进行fine-tune


## 运行代码
Fine-tune： 使用fine-tune文件夹中的fine-tune代码即可。有的是jupyter notebook形式的，有的模型提供了多种fine-tune代码:使用transformers trainer或使用pytorch training loop。  

Attack：直接使用各个攻击模块中的attack.sh即可运行，若要修改运行参数，请查看attack.py中args_parser  

其中CodeLLaMA和DeepSeek的causal模型使用vllm框架进行加速推理

## Baseline
Natural attack for pre-trained models of code  

https://dl.acm.org/doi/abs/10.1145/3510003.3510146



