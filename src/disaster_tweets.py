# -*- coding: utf-8 -*-
"""Disaster Tweets.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16vAkn9rBvm3G9SqM5wtyZ_Rv9CK6dELT
"""

!pip install datasets

!pip install emoji

!pip install contractions

"""**Disaster Tweets Sentiment Analysis Notebook**

This notebook is designed to analyze the sentiment of tweets related to disasters using a pre-trained model for sentiment classification. It includes the following steps:

**1. Import Libraries:**
"""

# Import necessary libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from torch import optim
from datasets import load_dataset
from sklearn.metrics import confusion_matrix

# Import custom functions from src directory
from src.preprocessing import preprocess_text
from src.utils import calculate_accuracy, tokenize_text
from src.prediction_utils import predict_sentiment

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/gdrive')

"""**2. Data Loading and Preprocessing**"""

# Load the dataset from a CSV file
df = pd.read_csv('data/train.csv')

df.head()

df.shape

# Display the distribution of target values in the dataset
df['target'].value_counts(normalize='true')

# Select relevant columns for further processing
data = df[['text', 'target']]

data.head()

# Apply text preprocessing to the 'text' column
data['text'] = data['text'].apply(preprocess_text)

data.head()

# Map the 'target' column to binary values (0 or 1)
data['target'] = data['target'].apply(lambda x: 1 if x==0 else 0)

data.head()

"""**3. Model Loading and Configuration**"""

# Load model configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract relevant model configuration parameters
model_name = config['model_config']['model_name']
num_labels = config['model_config']['num_labels']
epochs = config['training_config']['epochs']
learning_rate = config['training_config']['learning_rate']

# Determine the device to use (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model and tokenizer from Hugging Face Transformers
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the model for the specified number of labels
model.config.num_labels = num_labels

"""**4. Data Preparation for Training**"""

# Tokenize the text data and convert to PyTorch tensors
training_data = [tokenize_text(text, tokenizer, device) for text in data['text']]
training_labels = data['target']

"""**5. Model Training**"""

# Initialize the optimizer and loss function
optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
  outputs = []
  model.train()

  for text, label in zip(training_data, training_labels):

    output = model(**text)
    loss = criterion(output.logits, torch.tensor([label]).to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs.append(torch.argmax(output.logits).item())

  # Calculate and print accuracy for the current epoch
  accuracy = calculate_accuracy(outputs, training_labels)
  print(f'epoch: {epoch} has an Accuracy of {accuracy}')

"""**6. Sentiment Prediction on Entire Data**"""

# Make predictions on the entire dataset
predicted_sentiments = []
for text in data["text"]:
    encoded_text = tokenize_text(text, tokenizer, device)
    predicted_sentiment = predict_sentiment(model, encoded_text)
    predicted_sentiments.append(predicted_sentiment)

"""**7. Evaluation**"""

# Assuming `data['target']` contains true labels and `predicted_sentiments` contains predictions
result = confusion_matrix(data['target'], predicted_sentiments)
print(result)

"""**8. Saving the model**"""

model.save_pretrained('model')