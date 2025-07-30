#import os
import pandas as pd

from functions.preproc import preprocess_data
from functions.model_train import get_best_model

def main():

    """
    Build the pipeline for the dengue DS competition
    """

    # load the provided data
    train_features = pd.read_csv(
    "./data/raw/dengue_features_train.csv", index_col=[0, 1, 2]
    )

    train_labels = pd.read_csv(
    "./data/raw/dengue_labels_train.csv", index_col=[0, 1, 2]
    )

    print("-- Pipeline starts ---")

main()



