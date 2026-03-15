import pandas as pd
import numpy as np
import os
import argparse

def generate_dummy_traffic_data(output_path='data/STIN.csv', num_samples=5000, num_features=10):
    """
    Generate a dummy network traffic dataset for STELLAR out-of-the-box running.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate random features
    print(f"Generating {num_samples} dummy samples with {num_features} features...")
    features = np.random.randn(num_samples, num_features)
    
    # Generate random binary labels
    labels = np.random.choice(['BENIGN', 'DDoS'], size=num_samples, p=[0.7, 0.3])
    
    # Create DataFrame
    columns = [f'Feature_{i}' for i in range(num_features)]
    df = pd.DataFrame(features, columns=columns)
    df['Label'] = labels
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully saved dummy dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy traffic dataset for STELLAR.")
    parser.add_argument("--output", type=str, default="data/STIN.csv", help="Output path for the generated CSV.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to generate.")
    parser.add_argument("--features", type=int, default=10, help="Number of numeric features.")
    
    args = parser.parse_args()
    generate_dummy_traffic_data(args.output, args.samples, args.features)
