import torch
from torch.utils.data import Dataset
import pandas as pd
from src.utils import preprocess_text

class SentimentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Map labels to integers
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label_str = self.data.iloc[index]['label']
        
        # Preprocess text
        text = preprocess_text(text)
        
        # Encode text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        label = self.label_map.get(label_str, 1) # Default to Neutral if unknown
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
