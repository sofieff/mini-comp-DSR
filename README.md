# DengAI - Dengue Fever Epidemic Prediction

[![License](https://img.shields.io/github/license/your-username/dengai)](LICENSE)
[![Build 
Status](https://github.com/your-username/dengai/actions/workflows/main.yml/badge.svgStatus](https://github.com/your-username/dengai/actions/workflows/main.yml/adge.svg)](https://github.com/your-username/dengai/actions/workflows/main.yml)

## Description

https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/This repository contains a solution for the DengAI competition hosted on 
DrivenData.org. The goal is to predict the incidence of dengue fever in various 
regions using historical weather data. This project leverages the DengAI 
competition dataset and aims to provide a robust and accurate prediction model.

## Competition Link

[https://www.drivendata.org/dengai/](https://www.drivendata.org/dengai/)

## Dataset

The dataset consists of historical weather data for various regions, including 
temperature, rainfall, humidity, and more. The target variable is the number of 
dengue fever cases.

## Solution Overview

This solution implements a [briefly mention the core model(s) used, e.g.,  
"gradient boosting model", "deep learning model", "a combination of models"] to 
predict dengue fever incidence. 

Key components of the solution include:

* **Data Preprocessing:** Handling missing values, feature engineering, and data 
scaling.
* **Model Training:** Training the chosen model(s) on the historical data.
* **Model Evaluation:** Evaluating the model's performance using appropriate 
metrics (e.g., RMSE, MAE).
* **Prediction Generation:** Generating predictions for the test data.

## Dependencies

* Python 3.7+
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* [List any other specific libraries used, e.g., TensorFlow, PyTorch, etc.]

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dengai.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd dengai
   ```

3. **Configure the environment:**
   [Provide instructions on how to configure any necessary environment variables or 
API keys.]

4. **Run the prediction script:**
   ```bash
   python predict.py --data_path <path_to_data_folder> --output_path 
<path_to_output_folder>
   ```
   [Explain the arguments for the prediction script.]

## Evaluation

The performance of the solution is evaluated using [mention the evaluation metrics 
used in the competition]. The results are reported in the `results/` directory.

## Contributing

Contributions are welcome! Please feel free to submit pull requests for bug fixes, 
feature enhancements, or improvements to the code.

## License

This project is licensed under the [License Name] - see the [LICENSE](LICENSE) file 
for details.

## Authors
Kasia, Hamed, Sofia
