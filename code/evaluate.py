import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import SentimentDataset
from src.model import SentimentClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import argparse

def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
            
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    
    return predictions, prediction_probs, real_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='code/data')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    test_dataset = SentimentDataset(
        os.path.join(args.data_dir, 'test.csv'), 
        tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    model = SentimentClassifier(n_classes=3)
    if os.path.exists('best_model_state.bin'):
        model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
    else:
        print("Warning: No trained model found. Using random weights.")
        
    model = model.to(device)
    
    y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_loader,
        device
    )
    
    class_names = ['Negative', 'Neutral', 'Positive']
    print(classification_report(y_test, y_pred, target_names=class_names))
    
if __name__ == '__main__':
    main()
