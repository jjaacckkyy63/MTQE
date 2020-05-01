# Overview
Task: [Machine translation quality estimation](./Capstone_Proposal_QE.PDF)

Sentence-level Quality Estimation Shared Task of [WMT20](http://www.statmt.org/wmt20/)

# Datasets
1) QE data from WMT20: https://github.com/facebookresearch/mlqe
2) English-German and English-Chinese parallel data from [News-Commentary](http://opus.nlpl.eu/News-Commentary.php)

# Baseline model
1) [OpenKiwi tool](https://github.com/Unbabel/OpenKiwi/blob/master/kiwi/models/predictor_estimator.py) + 
   [mbert pretrained vectors](https://github.com/google-research/bert/blob/master/multilingual.md)
   
    a) [API page](https://unbabel.github.io/OpenKiwi/)
    
    b) [2019 QE + BERT/XLM](http://www.statmt.org/wmt19/pdf/54/WMT06.pdf)
    
    c) [2018 QEbrain transformer code](https://github.com/lovecambi/qebrain) <br>
       [2018 QEbrain transformer](https://www.aclweb.org/anthology/W18-6465.pdf) <br>
       [2018 QEbrain original transformer model](https://arxiv.org/pdf/1807.09433.pdf)    
       
    d) [2018 Automatic Post-eding](https://www.aclweb.org/anthology/W18-1804.pdf)    
    
    e) [2017 QE + Bilstm](https://dl.acm.org/doi/10.1145/3109480) <br>
       [2017 QE + Bilstm on WMT17 task](http://www.statmt.org/wmt17/pdf/WMT63.pdf)

# Submission Link:
https://competitions.codalab.org/competitions/24207

# Experiments:
1. Transformer-based predictor
2. NCE and NEG loss
3. Fine-tuned pretrained models provided by [WMT20](http://www.statmt.org/wmt20/quality-estimation-task.html)
4. Additional parallel data for en-de and en-zh pairs
5. Ensembles

# Usage:
1. All configurations can be set in the [Config](./config.py) file. Some important configurations includes:

   1) Trained model (Bilstmpredictor, Estimator, ...)
   2) Paths for saving and loading checkpoints
   3) Used language pairs
   4) Hyper-parameters (epochs, batch size, learning rate, ...)
  
2. All of the pipeline can be run with the [Main](./main.py) file, which includes:

   1) Train
   2) Predict
   3) Evaluate


