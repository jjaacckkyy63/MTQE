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

2)  slides:
https://docs.google.com/presentation/d/1scd7uLgS4FUexYf7AtmT33L59qtOMURi7YsKUrW0z8g/edit?fbclid=IwAR3MyWSaGAGfAW7G-gA3t6oCd5pi87LhwsLiEcHy3Q0-c8JIDR3Z-hj0YJs#slide=id.g7336ca79bc_0_47


# Possible Direction
1) Transfer learning from label data in other languages, including

    a) fine tuning on the dev set

    b) multi-task learning

    c) upsampling

    d) pretrained representation

    e) automatic label for two other languages

# Submission Link:
https://competitions.codalab.org/competitions/24207

# Apr. 17 Notes:
Tasks:
1. Embedding (XLM-R)
2. Predictor arch (change to transformer), Pre/Post QEFV extraction
3. Training objective (currently MLM)
4. Training method for few-shot learning - word2vec

7k training dataset:
https://github.com/facebookresearch/mlqe/tree/master/data


# Apr. 21 Notes:
TODO:
1. Check loading of vocabs
2. Use more data
   https://github.com/facebookresearch/mlqe?fbclid=IwAR3hEEWsklEGzm0qQ-FLD0_qFz5-VZEKijEhJtfVQhwAQDL_8TmGCRvpUUs
3. Use XLM-R as predictor (extract feature)
4. Draft report
5. Baseline: Predictor train w/ all five languages, Estimator train w/o de and zh datasets

Questions for FB:
1. What dataset is used for training (predictor and estimator)?

# Experiments:
1. Data
2. Pre-trained
3. Model - loss, (transformer, embedding)
