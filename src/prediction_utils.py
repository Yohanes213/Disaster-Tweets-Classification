import torch

def predict_sentiment(model, encoded_text):
    with torch.no_grad():
        outputs = model(**encoded_text)
        pred = torch.argmax(outputs.logits).item()
    return pred