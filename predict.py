import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import os

# Load the pre-trained model and tokenizer from the saved directory
model_dir = "./language_predictor_model"
model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)

# Mapping labels to languages (adjust if necessary based on your training labels)
label_to_language = {
    0: 'English',
    1: 'Spanish',
    2: 'French',
    3: 'German',
    4: 'Italian'
}

# Function to predict the language of a list of sentences
def predict_language(sentences):
    # Tokenize and prepare inputs for the model
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).tolist()
    
    # Map labels to language names and print results
    for sentence, label in zip(sentences, predicted_labels):
        language = label_to_language[label]
        print(f"Sentence: {sentence}")
        print(f"Predicted Language: {language}\n")

# Example sentences to test the predictor
example_sentences = [
    "This is a test sentence in English.",
    "Esta es una frase de prueba en español."
]

# Run predictions on example sentences
predict_language(example_sentences)
