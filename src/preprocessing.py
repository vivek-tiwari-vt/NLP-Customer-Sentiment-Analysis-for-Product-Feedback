# src/data_preprocessing.py
import pandas as pd
import re

def preprocess_fasttext_data(input_path, output_path):
    """
    Convert fastText format (__label__1/2 <text>) to DataFrame
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract label and text
            match = re.match(r'(__label__[12])\s(.*)', line)
            if match:
                label = match.group(1).replace('__label__', '')
                text = match.group(2).strip()
                data.append({
                    'text': text,
                    'sentiment': 'negative' if label == '1' else 'positive'
                })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df