---
layout: post
categories: English
title: Chinese word segmentation statistical approach(I)
tags: NLP word-segmentation
toc: false
date: 2022-05-25 22:42 +0300
math: true
pin: false
---
_**Language model** is a model predicting probabilities of sentences or sequences of words._    

This blog discuss about how to do word segmentation. Unlike European languages, some Asian languages like Chinese and Janpanese, don't have spaces between each words. Humans understand languages because we automatically have recognized words in heads. For machinese, those are merely sequences of symbols and characters. If we hope computers to correctly deal with them, word segmentation is a necessary step. Here the blog is going to discuss a statistical approach, N-gram.  

It starts with unigram (1-gram) which sees every word is independent from context. So with training data, the probability of a word is calculated as:  
$$ 
P(word)=\frac{C(word)}{N}
$$  
where N is the total number of words. But there would be some exceptions that some characters or words never show up in the corpus but it doesn't mean the word doesn't exist, does it? Therefore, we adopt plus 1 and V on numerator and denominator. The equation becomes:  
$$ 
P(word)=\frac{C(word)+1}{N+|V|}
$$  
where |V| is unique type of words since the smoothing way think every word should at least show up for once.
Since all preparing steps are done, now the task is to find the segmentation way with highest probabilities.

![r](https://raw.githubusercontent.com/goodeda/goodeda.github.io/main/assets/post_img/unigram.png)   

This could be done by greedy algorithm(local optimum) or other global optimum way.

This unigram method isolates each words and ignore contextual words. It simply select words according to their frequencies. If we use it as a model for language generation, the produced sentence would definitely quite fragmented and incoherent. Thus, it's not a very good solution for this task. Nevertheless, don't forget the N can be larger and more information is taken into account.  

Reference: [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/3.pdf)