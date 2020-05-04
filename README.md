# Author Classification with BERT
###### By Chloe Sheen, Tina Huang, Joseph Liu, Worthan Kwan & Mia Mansour

## Abstract
Authorship classification is an essential topic in Natural Language Processing, and it can be used in tasks such as identifying most likely authors of documents, plagiarism checking, and as a new way for recommending authors to readers based on the reader’s preferred style of writing. In this project, we explored this problem at different levels, with different deep learning models and with different implementations of combining BERT embeddings with bag of words in a neural network on the Reuters\_50\_50 dataset. Out of all of the models we tested, we achieved the best result of 92.9\% with sentence embeddings output by BERT as features as well as a bag-of-words that were fed into a simple forward-feed neural network, using an end-to-end embedding and classification method.

## How to run
All our code for this project are self-contained in individual Jupyter notebooks
which can be found in the `code/` folder. Please refer to the readme for more
detailed instructions.
