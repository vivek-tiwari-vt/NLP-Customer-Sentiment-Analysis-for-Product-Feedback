import pandas as pd
import re
import os
from typing import Optional

def preprocess_fasttext_data(input_path: str, output_path: str) -> Optional[pd.DataFrame]:
    """
    Extract and transform fastText format data into CSV.
    
    Args:
        input_path (str): Path to input file in fastText format (__label__1/2 <text>)
        output_path (str): Path where the CSV file will be saved
    
    Returns:
        pd.DataFrame: DataFrame with columns 'text' and 'sentiment'
        None: If file processing fails
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process the data
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Extract label and text using regex
                    match = re.match(r'(__label__[12])\s(.*)', line.strip())
                    if match:
                        label = match.group(1).replace('__label__', '')
                        text = match.group(2).strip()
                        
                        # Map labels to sentiments
                        sentiment = 'negative' if label == '1' else 'positive'
                        
                        data.append({
                            'text': text,
                            'sentiment': sentiment
                        })
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        # Create DataFrame and save to CSV
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Successfully processed {len(df)} records")
            print(f"Data saved to: {output_path}")
            return df
        else:
            print("No valid data was processed")
            return None
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def main():
    """Main function to run the preprocessing pipeline"""
    # Define paths
    input_path = 'data/raw/test.ft.txt'
    output_path = 'data/processed/test.csv'
    
    print("Starting data preprocessing...")
    df = preprocess_fasttext_data(input_path, output_path)
    
    if df is not None:
        print("\nFirst few rows of processed data:")
        print(df.head())
        print(f"\nDataset shape: {df.shape}")
        print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

if __name__ == "__main__":
    main()