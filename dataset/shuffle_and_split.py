"""
Dataset shuffling and splitting
"""
import pandas


class DatasetFractionsError(Exception):
    """
    Exception which is raised when shuffle_and_split method is called with invalid configuration of dataset fractions
    """
    pass


def shuffle_and_split(csv_fn, train_frac=0.85, dev_frac=0.10):
    """
    Loads the main dataset csv, that is given by csv_fn, splits the rows into 3 parts -
    train, dev and test. Then samples are shuffled and exported into 3 separate csv files.
    Fraction of train and dev examples is given and fraction of test examples is then inferred from the first
    mentioned parameters.

    :param csv_fn: filename of main dataset csv
    :param train_frac: fraction of train examples in the whole dataset
    :param dev_frac: fraction of dev examples in the whole dataset
    :return: N/A
    """
    if train_frac + dev_frac > 1.0:
        raise DatasetFractionsError(
            "Invalid fraction setup. Train and dev dataset must for less than the whole dataset.")

    df = pandas.read_csv(csv_fn)

    train_end_idx = int(len(df) * train_frac)
    dev_end_idx = train_end_idx + int(len(df) * dev_frac)

    df = df.sample(frac=1.0).reset_index(drop=True)

    df[:train_end_idx].to_csv("train_ds.csv", header=False, index=False)
    df[train_end_idx:dev_end_idx].to_csv("dev_ds.csv", header=False, index=False)
    df[dev_end_idx:].to_csv("test_ds.csv", header=False, index=False)

