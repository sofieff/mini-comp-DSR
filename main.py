#import os
import pandas as pd
import numpy as np

from functions.preproc import preprocess_data
from functions.model_train import get_best_model

def main():

    """
    Build the pipeline for the dengue DS competition
    """

    # load the provided data
    train_features = pd.read_csv(
    "./data/interim/dengue_train_new_features_select.csv", index_col=[0, 1, 2]
    )

    train_labels = pd.read_csv(
    "./data/raw/dengue_labels_train.csv", index_col=[0, 1, 2]
    )

    print("-- Pipeline starts ---")

    # calling preprocessing
    
    sj_train, iq_train = preprocess_data(
        "./data/interim/dengue_train_new_features.csv",
        labels_path="./data/raw/dengue_labels_train.csv",
    )
    print("-- preprocessing completed---")

    sj_train_subtrain = sj_train.head(800)
    sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

    iq_train_subtrain = iq_train.head(400)
    iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

    sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
    iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)


    print("-- model testing completed---")

    # submission
    sj_test, iq_test = preprocess_data("./data/interim/dengue_test_new_features.csv")

    sj_predictions = sj_best_model.predict(sj_test).astype(int)
    iq_predictions = iq_best_model.predict(iq_test).astype(int)

    submission = pd.read_csv("./data/processed/submission_format.csv", index_col=[0, 1, 2])

    submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
    submission.to_csv("./data/processed/benchmark_new_features_select.csv")

main()



