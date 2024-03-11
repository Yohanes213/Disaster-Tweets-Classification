# Import necessary libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from torch import optim
from datasets import load_dataset
from sklearn.metrics import confusion_matrix

# Import custom functions from src directory
from preprocessing import preprocess_text
from utils import calculate_accuracy, tokenize_text
from prediction_utils import predict_sentiment

# Function for training the model
def train_model(model, tokenizer, optimizer, criterion, device, training_data, training_labels, epochs):
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
        print(f'Epoch: {epoch} - Accuracy: {accuracy}')

    return model

# Function for making predictions
def make_predictions(model, tokenizer, device, data):
    predicted_sentiments = []
    for text in data["text"]:
        encoded_text = tokenize_text(text, tokenizer, device)
        predicted_sentiment = predict_sentiment(model, encoded_text)
        predicted_sentiments.append(predicted_sentiment)

    return predicted_sentiments

def main():
    # Load the dataset from a CSV file
    df = pd.read_csv('data/train.csv')

    # Select relevant columns for further processing
    data = df[['text', 'target']]

    # Apply text preprocessing to the 'text' column
    data['text'] = data['text'].apply(preprocess_text)

    # Map the 'target' column to binary values (0 or 1)
    data['target'] = data['target'].apply(lambda x: 1 if x == 0 else 0)

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

    # Tokenize the text data and convert to PyTorch tensors
    training_data = [tokenize_text(text, tokenizer, device) for text in data['text']]
    training_labels = data['target']

    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training the model
    trained_model = train_model(model, tokenizer, optimizer, criterion, device, training_data, training_labels, epochs)

    # Make predictions on the entire dataset
    predicted_sentiments = make_predictions(trained_model, tokenizer, device, data)

    # Assuming `data['target']` contains true labels and `predicted_sentiments` contains predictions
    result = confusion_matrix(data['target'], predicted_sentiments)
    print(result)

    # Save the trained model
    trained_model.save_pretrained('model')

if __name__ == "__main__":
    main()
