import pprint
import argparse
from sklearn.metrics import f1_score

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)


def readLabels(datafile):
    ''' Datafile contains 3 columns,
        col1: hyponym
        col2: hypernym
        col3: label
    '''

    labels = []
    with open(datafile, 'r') as f:
        inputlines = f.read().strip().split('\n')

    for line in inputlines:
        labels.append(line)
    return labels


def computePRF(truthlabels, predlabels):
    correct = 0
    incorrect = 0
    for t, p in zip(truthlabels, predlabels):
        if t == p:
            correct += 1
        else:
            incorrect += 1

    accuracy = float(correct)/float(correct+incorrect)
    print("Accuracy:{}".format(accuracy))

    f1 = f1_score(truthlabels, predlabels, average='micro')
    print("F1:{}".format(f1))
    return accuracy, f1
    # prec = float(tp)/float(tp + fp)
    # recall = float(tp)/float(tp + fn)
    # f1 = 2*prec*recall/(prec + recall)

    # print("Precision:{} Recall:{} F1:{}".format(prec, recall, f1))

    # return prec, recall, f1


def main(args):
    gold = readLabels(args.goldfile)
    pred = readLabels(args.predfile)
    print("Performance")
    computePRF(gold, pred)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
