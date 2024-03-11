# Diaster Tweets Classification

## Overview

This project focuses on classifying disaster-related tweets using a pre-trained transformer-based model. The goal is to predict whether a given tweet is related to a disaster or not. The project includes data preprocessing, data exploration, model training, and evaluation.

## Project Structure

The project is organized into the following directories and files:

- **data/:** Contains the dataset used for training (`train.csv`).
- **src**: Source code directory.
  - **preprocessing.py**: Module for text preprocessing functions.
  - **utils.py**: Module for utility functions.
  - **prediction_utils.py**: Module for prediction-related functions.
  - **train.py**: Script for training the machine learning model.
- **notebooks/:**
  - `sentiment_analysis_training.ipynb:` Jupyter notebook for model training.
- **models/:** Placeholder for saving trained models.
- **config.json:** Configuration file specifying model parameters.
- **requirements.txt:** List of Python dependencies.

## Getting Started

1. Clone the repository:
   ``` bash
   git clone https://github.com/Yohanes213/Disaster-Tweets-Classification.git
   cd Disaster-Tweets-Classification
    ```

2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt
    ```
   
3. Set up the necessary data and model configuration:
    - Place the training dataset (`train.csv`) in the `data/` directory.
    - Configure model parameters in `config.json`.

4. Run the training script to train the machine learning model:
    ``` bash
    python src/train.py
    ```

## Custom Functions

Custom Python functions are organized into separate modules within the `src/` directory:

- `preprocessing.py:` Contains the `preprocess_text` function for text preprocessing.
- `utils.py:` Includes the `calculate_accuracy` and `tokenize_text` functions.
- `prediction_utils.py:` Holds the `predict_sentiment` function for making predictions.

## Results

The model's performance is evaluated using metrics such as accuracy and a confusion matrix. Results can be found in the `train_model.ipynb` notebook.

## Contributing

Contributions are welcome! If you find issues or have suggestions, please open an issue or create a pull request.

   
