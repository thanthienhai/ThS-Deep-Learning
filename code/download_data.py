import pandas as pd
import os
import gdown

def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists. Skipping.")

def process_split(data_dir, split_name, urls):
    sentences_path = os.path.join(data_dir, f'{split_name}_sentences.txt')
    sentiments_path = os.path.join(data_dir, f'{split_name}_sentiments.txt')
    
    download_file(urls['sentences'], sentences_path)
    download_file(urls['sentiments'], sentiments_path)
    
    with open(sentences_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
        
    with open(sentiments_path, 'r', encoding='utf-8') as f:
        sentiments = [int(line.strip()) for line in f.readlines()]
        
    df = pd.DataFrame({'text': sentences, 'label_id': sentiments})
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df['label'] = df['label_id'].map(label_map)
    
    return df[['text', 'label']]

def download_and_process_data(output_dir='code/data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # URLs extracted from the original dataset script
    # Fixed typo in train sentiments URL
    urls = {
        'train': {
            "sentences": "https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download",
            "sentiments": "https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download", 
        },
        'validation': {
            "sentences": "https://drive.google.com/uc?id=1sMJSR3oRfPc3fe1gK-V3W5F24tov_517&export=download",
            "sentiments": "https://drive.google.com/uc?id=1GiY1AOp41dLXIIkgES4422AuDwmbUseL&export=download",
        },
        'test': {
            "sentences": "https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download",
            "sentiments": "https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download",
        }
    }

    for split in ['train', 'test', 'validation']:
        print(f"Processing {split} split...")
        df = process_split(output_dir, split, urls[split])
        output_path = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} examples to {output_path}")
        
    print("Done! Dataset downloaded and processed.")

if __name__ == "__main__":
    download_and_process_data()
