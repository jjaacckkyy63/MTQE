# Overview

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

    b) [paper](http://www.statmt.org/wmt19/pdf/54/WMT06.pdf)

# Possible Direction
1) Transfer learning from label data in other languages, including

    a) fine tuning on the dev set

    b) multi-task learning

    c) upsampling

    d) pretrained representation

    e) automatic label for two other languages

# Submission Link:
https://competitions.codalab.org/competitions/24207
