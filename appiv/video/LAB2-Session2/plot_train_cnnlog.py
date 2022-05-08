"""
Plot a Given a training log file.
"""
import csv
import matplotlib.pyplot as plt


def plot_accuracy(training_log, n_classes):
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

        plt.plot(accuracies_t, label="CNN train accuracy")
        plt.plot(accuracies_v, label="CNN valid accuracy")
        plt.plot(cnn_benchmark, label="Random accuracy")
        plt.title(f"CNN Accuracies ({n_classes} classes)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()


        file_name = training_log.split('/')[-1]
        file_name = file_name.split('log')[0]

        plt.savefig(f"{file_name}png")
        plt.show()
        


#if __name__ == '__main__':    
#    training_log = 'data/logs/inception-training-1602753951.8033223.log'
#    main(training_log)
