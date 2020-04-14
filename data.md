# Dataset

## Reuters 50 50 (C50)
Reuters 50 50 is a subset of RCV1, which is a collection of over 800,000 English 
language news stories dating from August 20, 1996 to August 19, 1997 that have 
been made available by Reuters, Ltd. 

Each story is categorised into one of four hierachical categories:
* Corporate/Industrial (CCAT)
* Economics (ECAT)
* Government/Social (GCAT)
* Markets (MCAT)

(A full list of sub-categories can be found 
[here](https://gist.github.com/gavinmh/6253739))

The Reuters 50 50 dataset is a subset containing 5000 stories by the top 50
authors according to the number of stories written. We chose
this dataset because it has already been used by previous author identification
studies [1]..

The authors and stories were selected so that at least one subtopic of the class
CCAT was included to minimise the topic factor in distinguishing among texts.
The train/test set is split 50:50 with  each set consisting of 2,500 texts (50 
per author). However, in the paper by Qian et al [1], they re-organised it into 
a 9:1 train/test split. In order to benchmark our results, we will also split
our dataset accordingly. 


[[1]]: C Qian, T He, R Zhang. Deep Learning based Authorship Identification,
2018

[1]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760185.pdf
