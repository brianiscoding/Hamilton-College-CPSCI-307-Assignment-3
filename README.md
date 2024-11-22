# Project Title

Hamilton College CPSCI 307 Assignment 3

## Description

This is for Hamilton College CPSCI 307 Assignment 3. Our group is made up of Connor, Tyrone, Brian. This project is a deep learning model that translates english to french, using a seq2seq transformer model based on transformer architecture.

## Dependencies

The following libraries must be installed:
- torch
- nltk
- sentencepiece
- pytorch_lightning
- sklearn

The following files must be in the directory:
- inference.py
- train.py
- checkpoint.ckpt
- english_bpe.model
- french_bpe.model
- eng_fra.txt

## Executing program

To execute the program, run the following line:

python inference.py checkpoint.ckpt english_bpe.model french_bpe.model --input [*input string here*]

In place of the brackets, include a english sentence (as a string) of what you would like to be translated. The file will then output its french translation.

## Authors

Brian Tran

Connor Whynott

Tyrone Xue
