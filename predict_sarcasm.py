# import necessary libraries
import numpy as np
import pandas as pd
import argparse
import pickle

def predict_sarcasm_values(df: pd.DataFrame) -> pd.DataFrame:
    # import trained model weights into ensemble model
    
    
    return None

def main():
    # parser to get arguments from command line
    parser = argparse.ArgumentParser(description="Run ensemble model predictions.")
    
    parser.add_argument("--input", type=str, required=True, help="Name of input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions")
    
    args = parser.parse_args()
    
    print(f"- Reading data from file: {args.input}")
    print(f"- Outputting to file: {args.output}\n")
    
    print("=" * 50)
    print("Getting dataset (ensure that the input dataset is located within the same folder as this file)")
    print("=" * 50)
    
    df = pd.read_csv(args.input)
    print("...Input dataset has been retrieved")
    
    
    
    return
    
if __name__ == "__main__":
    main()
