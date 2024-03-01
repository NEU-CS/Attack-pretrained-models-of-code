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
已经fine-tune好的部分模型在huggingface上（使用trainer训练的）。在Huggingface上models搜索"ljcnju"即可  
如https://huggingface.co/ljcnju/DeepSeek7bForCodeTrans

## 运行代码
Fine-tune： 使用fine-tune文件夹中的fine-tune代码即可。有的是jupyter notebook形式的，有的模型提供了多种fine-tune代码:使用transformers trainer或使用pytorch training loop。  

Attack：直接使用各个攻击模块中的attack.sh即可运行，若要修改运行参数，请查看attack.py中args_parser  

其中CodeLLaMA和DeepSeek的causal模型使用vllm框架进行加速推理

## Baseline
Natural attack for pre-trained models of code  

https://dl.acm.org/doi/abs/10.1145/3510003.3510146


## 实验结果
本项目研究各种预训练模型在代码处理任务上的鲁棒性，通过扩展beam search和死代码插入进一步提升攻击成功率  
Metrics:
### Attack success rate(ASR)
针对于分类任务，即被攻击模型(Victim model)在原本预测正确的测试用例上，经过攻击后，有多少样本无法预测正确  
注意ASR是一个加权攻击成功率，本项目只研究Victim model的Robustness，不研究Performance。Performance越高的Victim model被攻击的样本就越多。  
因此，加权的ASR = 原本的ASR / Performance（即accuracy）  

ASR(Attack success rate)越高越好  

### CodeBLEU decline(CD)
针对于生成式任务：被攻击模型（Victim model）在测试集上的平均CodeBLEU下降程度。其中CodeBLEU是一种衡量代码生成质量的指标，它综合了代码具有的基础的文本信息，代码语法信息以及其他的数据流信息匹配程度，相比于BLEU更适合于衡量代码的生成质量。  
同样的，CodeBLEU decline也是一种加权的衡量指标，以消除Models之间Performance差异的影响  

### Model query times(MQT)
即完成所有样本的攻击，对Victim model的总Query次数。由于本攻击是黑盒攻击，只能通过反复query model来进行攻击。Model query times就衡量了某种攻击方法的效率  
显然，Model query times越小越好

### Sccuess query rate(SQR)
指每一次query能带来多少成功攻击的价值，SQR(Sccuess query rate) = ASR或者CD/MQT  
这个指标综合了攻击质量和攻击效率。显然，SQR越高越好  
由于MQT通常是以万为数量级的，而ASR和CD是在0~2之间的，为了看起来方便，SQR = ASR(CD) * 10000 /MQT   


#### 分类任务（Authorship-Attribution）:


| Task                   | Model             | Performance                       |     | Baseline(Only replace and beam size = 1) | DeadCodeInsert(beam size = 1) | Mix(replace and insert;beam size = 1) |
|:----------------------:|:-----------------:|:---------------------------------:|:---:|:----------------------------------------:|:-----------------------------:|:-------------------------------------:|
| Authorship Attribution | CodeBert          | 0.817                             | ASR | 0.492044064                              | 0.652386781                   | 0.675642595                           |
|                        | CodeT5            | 0.809                             | ASR | 0.478368356                              | 1.143386897                   | 1.107540173                           |
|                        | CodeLLaMA-7b      | 0.748                             | ASR | 0.723262032                              | 1.282085561                   | 1.24197861                            |
|                        | DeepSeek-coder-7b | 0.885                             | ASR | 0.311864407                              | 1.091525424                   | 1.062146893                           |


上表是所有模型使用replace identifiers name,dead code insertion and mix attack的在分类任务上的ASR结果：  


    Performance：DeepSeek > CodeBert > CodeT5 > CodeLLaMA   


通过比较同一种攻击手段对不同Victim model的ASR可以得出不同Victim model的Robustness结果：  
通过观察列Baseline(Only replace and beam size = 1)可以知道，使用replace attack时:  

    Robustness: DeepSeek > CodeT5 > CodeBert > CodeLLaMA    

而使用insert attack时：  

    Robustness：CodeBert >> DeepSeek > CodeT5 > CodeLLaMA    

Mix attack时：  

    Robustness: CodeBert >> DeepSeek > CodeT5 > CodeLLaMA   
  

可以知道，带有Transformers decoder的模型在理解分类任务上，很难抵御住死代码插入这一类攻击  
Encoder-only式的CodeBert模型Robustness强很多  
本任务是对读取一段Python代码，预测完成这段代码的作者。数据集由Google完成  
由于每个人写代码都具有显著的个人风格，模型可以通过这些个人风格分辨出不同作者  
但DeadCode Insert攻击中，插入的死代码都是相同的，不具有任何一个人的代码风格  
因此，插入的死代码可能会平均掉每段代码的风格特征，从而使得各个模型更难分辨出作者  
而Replace attack是替换掉identifiers name，这也会影响到一段代码的个人风格特征  
但不如DeadCode Insert的影响大。因此，在本任务中，DeadCode Insert的攻击效果非常好  
然而为什么除CodeBert以外的模型对死代码这么敏感，我目前也没有明确的结论  

| Task                   | Model             | Performance |     | Baseline(Only replace and beam size = 1) | DeadCodeInsert(beam size = 1) | Mix(replace and insert;beam size = 1) |
|:----------------------:|:-----------------:|:-----------:|:---:|:----------------------------------------:|:-----------------------------:|:-------------------------------------:|
| Authorship Attribution | CodeBert          | 0.817       | MQT | 29996                                    | 2715                          | 10828                                 |
|                        |                   |             | SQR | 0.164036559                              | 2.402897904                   | 0.623977276                           |
|                        | CodeT5            | 0.809       | MQT | 23750                                    | 1837                          | 5720                                  |
|                        |                   |             | SQR | 0.201418255                              | 6.224207389                   | 1.936259044                           |
|                        | CodeLLaMA-7b      | 0.748       | MQT | 13578                                    | 1584                          | 3648                                  |
|                        |                   |             | SQR | 0.532671993                              | 8.093974504                   | 3.404546627                           |
|                        | DeepSeek-coder-7b | 0.885       | MQT | 19255                                    | 1857                          | 4487                                  |
|                        |                   |             | SQR | 0.161965415                              | 5.877896735                   | 2.367164905                           |

上表是不同攻击手段的MQT和SQR在分类任务上的结果  
可以看出DeadCodeInsert效率显著高于Replace，这与我们上面的分析一致  

| Task                   | Model             |     | Baseline(Only replace and beam size = 1) | Beam size = 2(only replace) | Beam size = 4(only replace) | Beam size = 8(only replace) |
|:----------------------:|:-----------------:|:---:|:----------------------------------------:|:---------------------------:|:---------------------------:|:---------------------------:|
| Authorship Attribution | CodeBert          | ASR | 0.492044064                              | 0.481028152                 | 0.492044064                 | 0.515299878                 |
|                        |                   | MQT | 29996                                    | 44714                       | 84979                       | 164961                      |
|                        |                   | SQR | 0.164036559                              | 0.107578868                 | 0.057901842                 | 0.031237679                 |
|                        | CodeT5            | ASR | 0.478368356                              | 0.478368356                 | 0.501854141                 | 0.501854141                 |
|                        |                   | MQT | 23750                                    | 43476                       | 81865                       | 158444                      |
|                        |                   | SQR | 0.201418255                              | 0.110030443                 | 0.06130265                  | 0.031673913                 |
|                        | CodeLLaMA-7b      | ASR | 0.723262032                              | 0.79144385                  | 0.846256684                 | 0.89973262                  |
|                        |                   | MQT | 13578                                    | 22762                       | 40196                       | 73350                       |
|                        |                   | SQR | 0.532671993                              | 0.347704002                 | 0.210532562                 | 0.122662934                 |
|                        | DeepSeek-coder-7b | ASR | 0.311864407                              | 0.350282486                 | 0.379661017                 | 0                           |
|                        |                   | MQT | 19255                                    | 34291                       | 63308                       |                             |
|                        |                   | SQR | 0.161965415                              | 0.102149977                 | 0.059970465                 | #DIV/0!                     |

上表是beam search不同beam size的对比结果  
beam search只使用了replace attack，完善dead code insert attack是我们未来的工作    
  

通过上表可以看出，beam search能够提升一定的ASR,特别是当beam size比较大的时候，提升会比较明显  
对于Beam size=2，CodeBert上效果反而下降了，我们认为这是一个偶然。因为我们的beam search设计时，beam size大的搜索并不一定会完全覆盖beam size小的搜索  
并不是beam size=2时，一条路径和beam size=1时一模一样，另一条不一样。而是两条路径都可能不一样，我们认为这样会扩大beam search的搜索范围。否则，限定了一条或多条   
路径，搜索范围一定会减小，我们认为这样的效果会更不好（虽然后者的设计一定使得beam size大的效果大于等于beam size小的）  
  

我们注意到，MQT和Beam size基本呈线性增长，这是符合预期的  
如果不在意攻击效率，可以选择更大的beam size以获得更好的ASR  
目前的beam size可以设置的最大值是每个identifiers的subtitute candidates数量  


#### 生成任务（CodeTrans）：  






因此，设计不同攻击手段（例如replace或insert），需要根据任务而定。不同的任务，Victim model获取到的特征信息都不一样，我们认为应该以任务为导向设计攻击手段，以获得更好的攻击效果  
