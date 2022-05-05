"""
Try to "classify" samples based on random chance vs always guessing
the same category.
"""
import random
from data import DataSet

import argparse

##ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam
parser = argparse.ArgumentParser()
parser.add_argument("-op", "--option", type=str, help="Action parameter")
args = parser.parse_args()
fix_mode = args.option

class_limit = 5  # int, can be 1-101 or None
seq_length = 5

data = DataSet(seq_length, class_limit)
nb_classes = len(data.classes)

# Try a random guess.
nb_random_matched = 0
nb_mode_matched = 0
for item in data.data:
    choice = random.choice(data.classes)
    actual = item[1]

    if choice == actual:
        nb_random_matched += 1

    if actual == fix_mode:
        nb_mode_matched += 1

random_accuracy = nb_random_matched / len(data.data)
mode_accuracy = nb_mode_matched / len(data.data)
print("Randomly matched %.2f%%" % (random_accuracy * 100))
print("Mode matched %.2f%%" % (mode_accuracy * 100))
