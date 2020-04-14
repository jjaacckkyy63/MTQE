# Overview
Task: [Machine translation quality estimation](./Capstone_Proposal_QE.PDF)

1. Continue score
2. 1000 language pair
3. Baseline - evaulate with pretrained lingual-bert / RNN / 1k sentence
4. Can be considered as few shot learning, avoiding using too much dev set, fucos on transfer learning from other languages.
5. If collecting other dataset, the score may not solid. Try to avoid using other German and Chinese language.
6. Allowing unlabel parallel data to do the pretrained embedding. 
7. Testing platform - 1k sentence in coda lab

# Q&A
1. Take the German and Chinese as few shot learning.

# Baseline model
1) [OpenKiwi tool](https://github.com/Unbabel/OpenKiwi/blob/master/kiwi/models/predictor_estimator.py) + 
   [mbert pretrained vectors](https://github.com/google-research/bert/blob/master/multilingual.md)

    a) [API page](https://unbabel.github.io/OpenKiwi/)

    b) [2019 QE + transformer](http://www.statmt.org/wmt19/pdf/54/WMT06.pdf)

    c) [2018 QEbrain](https://github.com/lovecambi/qebrain) <br>
        https://www.aclweb.org/anthology/W18-6465.pdf
    
    d) [2017 Quality estimator paper](https://dl.acm.org/doi/10.1145/3109480)

    e) [2017 Quality estimator paper on WMT17 task](http://www.statmt.org/wmt17/pdf/WMT63.pdf)

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
