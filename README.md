# Overview
This is the final project for Georgia Tech CS 7643: Deep Learning Course.

Task: [Machine translation quality estimation](./Capstone_Proposal_QE.PDF)

Sentence-level Quality Estimation Shared Task of [WMT20](http://www.statmt.org/wmt20/)

**Acknowledgement:** <br>
Thanks to Facebook AI collaborators in this course for sharing the project topic and guidelines.

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

# Possible Direction
1) Transfer learning from label data in other languages, including

    a) fine tuning on the dev set

    b) multi-task learning

    c) upsampling

    d) pretrained representation

    e) automatic label for two other languages

# Submission Link:
https://competitions.codalab.org/competitions/24207
