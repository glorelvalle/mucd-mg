import random
from data import DataSet

def random_vs_fixed(fix_mode="ApplyEyeMakeup", verbose_fix=True, verbose_random=False):
    """
    Classification of samples based on random chance vs always guessing the same category.
    """

    class_limit = 5  # int, can be 1-101 or None
    seq_length = 5

    data = DataSet(seq_length, class_limit)

    # Try a random guess
    nb_random_matched = 0
    nb_mode_matched = 0
    for item in data.data:
        choice = random.choice(data.classes)
        actual = item[1]

        if choice == actual:
            nb_random_matched += 1
        if actual == fix_mode:
            nb_mode_matched += 1

    random_accuracy = nb_random_matched/len(data.data)*100
    mode_accuracy = nb_mode_matched/len(data.data)*100

    if verbose_fix:
        print(f"- {fix_mode} mode matched %.2f%%" % (mode_accuracy))
    if verbose_random:
        print(f"- Random mode matched %.2f%%" % (random_accuracy))

    return random_accuracy, mode_accuracy
