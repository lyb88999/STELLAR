import pandas as pd
import numpy as np
import os
from data_simulator.cicids2017_generator import CICIDS2017Generator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def create_mock_cicids2017_csv(filename, num_rows=100):
    """Create a dummy CICIDS2017 CSV file for testing."""
    print(f"Creating mock CSV: {filename}")
    
    # Common CICIDS2017 columns
    columns = [
        " Destination Port", " Flow Duration", " Total Fwd Packets", 
        " Total Backward Packets", "Total Length of Fwd Packets", 
        " Total Length of Bwd Packets", " Fwd Packet Length Max", 
        " Fwd Packet Length Min", " Fwd Packet Length Mean", 
        " Fwd Packet Length Std", "Bwd Packet Length Max", 
        " Bwd Packet Length Min", " Bwd Packet Length Mean", 
        " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s", 
        " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", 
        " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean", 
        " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min", 
        "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std", 
        " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", 
        " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags", 
        " Fwd Header Length", " Bwd Header Length", "Fwd Packets/s", 
        " Bwd Packets/s", " Min Packet Length", " Max Packet Length", 
        " Packet Length Mean", " Packet Length Std", " Packet Length Variance", 
        "FIN Flag Count", " SYN Flag Count", " RST Flag Count", 
        " PSH Flag Count", " ACK Flag Count", " URG Flag Count", 
        " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio", 
        " Average Packet Size", " Avg Fwd Segment Size", " Avg Bwd Segment Size", 
        " Fwd Header Length.1", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", 
        " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", 
        "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes", 
        " Subflow Bwd Packets", " Subflow Bwd Bytes", "Init_Win_bytes_forward", 
        " Init_Win_bytes_backward", " act_data_pkt_fwd", " min_seg_size_forward", 
        "Active Mean", " Active Std", " Active Max", " Active Min", 
        "Idle Mean", " Idle Std", " Idle Max", " Idle Min", " Label"
    ]
    
    data = np.random.rand(num_rows, len(columns)-1)
    
    # Inject some "Infinity" strings to test cleaning
    data[0, 14] = np.nan # Simulate NaN
    
    df = pd.DataFrame(data, columns=columns[:-1])
    
    # Add "Infinity" string manually (pandas might treat it as object column)
    df["Flow Bytes/s"] = df["Flow Bytes/s"].astype(object)
    df.at[5, "Flow Bytes/s"] = " Infinity"
    
    # Add Labels
    labels = ["BENIGN"] * (num_rows // 2) + ["DDoS"] * (num_rows // 4) + ["PortScan"] * (num_rows - num_rows // 2 - num_rows // 4)
    # Shuffle labels slightly
    np.random.shuffle(labels)
    df[" Label"] = labels
    
    df.to_csv(filename, index=False)
    print("Mock CSV created.")

def test_cicids2017_generator():
    mock_file = "mock_cicids2017.csv"
    create_mock_cicids2017_csv(mock_file)
    
    try:
        # Initialize Generator
        # Using dummy satellite numbers
        generator = CICIDS2017Generator(num_satellites=10, num_orbits=2, satellites_per_orbit=5)
        
        # Load Data
        feature_dim, num_classes = generator.load_and_preprocess_data(mock_file)
        
        print(f"\nSUCCESS: Data loaded.")
        print(f"Feature Dim: {feature_dim}")
        print(f"Num Classes: {num_classes}")
        print(f"Labels found: {generator.label_encoder.classes_}")
        
        # Verify shape
        # We dropped 1 row with NaN and maybe 1 with Infinity (combined_df.dropna)
        # 100 rows -> ~98 rows expected
        print(f"Tensor shape X: {generator.X_data.shape}")
        
    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(mock_file):
            os.remove(mock_file)

if __name__ == "__main__":
    test_cicids2017_generator()
