import os
import pandas as pd

# link function .py files from functions/
from functions.preprocessing import funcitonname, funcitonname
from functions.mode_train import funcitonname, funcitonname

def pipeline(
    features_train_path: str,
    features_test_path: str,
    labels_train_path: str,

    ouput_predictions_path: str,
    ouput_model_path: str,

):

    """
    Build the pipeline for the dengue DS competition
    """

    print("Pipeline starts")

    # load data



