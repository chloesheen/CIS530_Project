# Scoring
## Mean F1 Scores

To quantify how good a system's answers are in terms of comparing the system's output to a corresponding set of gold-standard answers, some common metrics of evaluation include accuracy, Matthews correlation coefficient, precision, recall, and F1-score [1]. To avoid the problem of the accuracy paradox [2], where classification accuracy is not necessarily the best way of determining the best-performing model, we will be using F1 scores (2*((precision * recall)/(precision + recall))) to convey the balance between the precision and the recall. [Here](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)) is a breakdown of precision, recall, and F1-score. 

## Running the scoring script

This is how to run our script scoring.py from the command line: 

INSERT COMMANDS HERE

[[1]](https://web.stanford.edu/class/cs224n/reports/custom/15785631.pdf) Ma, G. (2019). Tweets Classification with BERT in the Field of Disaster Management.

[[2]](https://www.utwente.nl/en/eemcs/trese/graduation_projects/2009/Abma.pdf) Abma, B. J. M. (10 September 2009), Evaluation of requirements management tools with support for traceability-based change impact analysis (PDF), University of Twente, pp. 86–87