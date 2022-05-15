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
        plt.tight_layout()

        file_name = training_log.split('/')[-1]
        file_name = file_name.split('log')[0]

        plt.savefig(f"{file_name}pdf")
        plt.show()

def plot_loss(training_log, n_classes):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        loss_t = []
        loss_v = []
        top_5_accuracies = []
        cnn_benchmark = []  # random results
        for epoch, acc, loss, val_acc, val_loss, in reader:
            loss_t.append(float(loss))
            loss_v.append(float(val_loss))

        plt.plot(loss_t, label="CNN train loss")
        plt.plot(loss_v, label="CNN valid loss")
        plt.title(f"CNN Losses ({n_classes} classes)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()

        file_name = training_log.split('/')[-1]
        file_name = file_name.split('log')[0]
        file_name = file_name + '_loss.'


        plt.savefig(f"{file_name}pdf")
        plt.show()
        


if __name__ == '__main__':    
    training_log = ['logs/5-inception-training-1652454036.4590917.log',
                    'logs/10-inception-training-1652463027.015716.log',
                    'logs/15-inception-training-1652463027.015716.log',
                    'logs/20-inception-training-1652463027.015716.log']
    epochs = [5, 10, 15, 20]

    for file, e in zip(training_log, epochs):
        plot_loss(file, e)
#    main(training_log)

