"""
Plot a Given a training log file.
"""
import csv
import matplotlib.pyplot as plt


def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies_t = []
        accuracies_v = []
        top_5_accuracies = []
        cnn_benchmark = []  # random results
        for epoch, acc, loss, val_acc, val_loss, in reader:
            accuracies_t.append(float(acc))
            accuracies_v.append(float(val_acc))
            cnn_benchmark.append(0.2)  # random

        plt.plot(accuracies_t)
        plt.plot(accuracies_v)
        plt.plot(cnn_benchmark)
        plt.savefig('inception-training-1602753951.8033223.png')
        plt.show()
        


if __name__ == '__main__':
    
    training_log = 'data/logs/inception-training-1602753951.8033223.log'
    main(training_log)
