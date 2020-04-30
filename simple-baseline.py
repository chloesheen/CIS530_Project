# simple baseline 

import os  
import pandas as pd
import numpy as np 

train_dir = os.path.abspath("datasets/C50/C50train/")
test_dir = os.path.abspath("datasets/C50/C50test/") 
pred_file = 'pred.txt'

def readData(path): 
    '''
    get texts and corresponding authors
    '''
    authors = []
    texts = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".txt"):
                fp = os.path.join(root, name)
                author = root.split("/")[-1]
                authors.append(author)
                # read text
                with open(fp, 'r') as f:
                    text = f.read()
                    texts.append(text)

    return authors, texts


def write_baseline(test_texts, test_y):
    with open(pred_file, 'w') as f:
        for k in range(len(test_texts)):
            f.write(test_y[0] + '\n')

if __name__ == '__main__':
    train_y, train_texts = readData(train_dir)
    test_y, test_texts = readData(test_dir)
    write_baseline(test_texts, test_y)
