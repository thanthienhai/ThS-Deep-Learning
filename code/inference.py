import torch
from transformers import AutoTokenizer
from src.model import SentimentClassifier
from src.utils import preprocess_text
import argparse

def predict_sentiment(text, model, tokenizer, device):
    processed_text = preprocess_text(text)
    encoded_text = tokenizer.encode_plus(
        processed_text,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        
    class_names = ['Negative', 'Neutral', 'Positive']
    return class_names[prediction.item()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Text to predict')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    model = SentimentClassifier(n_classes=3)
    try:
        model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
    except:
        print("Warning: Could not load 'best_model_state.bin'. Using random weights.")
        
    model = model.to(device)
    model.eval()
    
    if args.text:
        sentiment = predict_sentiment(args.text, model, tokenizer, device)
        print(f"Text: {args.text}")
        print(f"Sentiment: {sentiment}")
    else:
        print("Enter text to predict (type 'quit' to exit):")
        while True:
            text = input("> ")
            if text.lower() == 'quit':
                break
            sentiment = predict_sentiment(text, model, tokenizer, device)
            print(f"Sentiment: {sentiment}")

if __name__ == '__main__':
    main()
