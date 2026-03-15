import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from data_simulator.real_traffic_generator import RealTrafficGenerator
import logging

class CICIDS2017Generator(RealTrafficGenerator):
    """
    CICIDS2017 Dataset Generator for network intrusion detection.
    Handling specific preprocessing for CICIDS2017 (Infinity, Labels, etc.)
    """
    def __init__(self, num_satellites: int, num_orbits: int, satellites_per_orbit: int):
        super().__init__(num_satellites, num_orbits, satellites_per_orbit)
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess_data(self, data_path: str, test_size: float = 0.2):
        """
        Load and preprocess CICIDS2017 data (single CSV or directory of CSVs).
        
        Args:
            data_path: Path to the CSV file or directory containing CSVs
            test_size: Proportion of dataset to include in the test split
            
        Returns:
            Tuple: (feature_dim, num_classes)
        """
        import os
        import glob
        
        csv_files = []
        if os.path.isdir(data_path):
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            print(f"Found {len(csv_files)} CSV files in directory: {data_path}")
        else:
            csv_files = [data_path]
            
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_path}")

        print(f"Loading CICIDS2017 data from {len(csv_files)} files...")
        
        combined_df = pd.DataFrame()
        chunk_size = 100000
        
        try:
            for csv_file in csv_files:
                print(f"Processing {os.path.basename(csv_file)}...")
                # Read a sample to clean columns
                # CICIDS2017 often has encoding issues, cp1252 is safer than utf-8
                # on_bad_lines='skip' handles malformed rows
                df_sample = pd.read_csv(csv_file, nrows=1, encoding='cp1252')
                df_sample.columns = df_sample.columns.str.strip()
                
                # Identify numeric columns (reuse logic or just read all)
                # We'll read in chunks
                csv_chunks = []
                try:
                    for chunk in pd.read_csv(csv_file, chunksize=chunk_size, encoding='cp1252', on_bad_lines='skip', low_memory=False):
                        chunk.columns = chunk.columns.str.strip()
                        csv_chunks.append(chunk)
                except TypeError:
                     # Fallback for older pandas versions using error_bad_lines
                     for chunk in pd.read_csv(csv_file, chunksize=chunk_size, encoding='cp1252', error_bad_lines=False, low_memory=False):
                        chunk.columns = chunk.columns.str.strip()
                        csv_chunks.append(chunk)
                
                file_df = pd.concat(csv_chunks, ignore_index=True)
                combined_df = pd.concat([combined_df, file_df], ignore_index=True)
                
            print(f"Loaded raw combined data shape: {combined_df.shape}")

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

        # 1. Handle Infinity and NaNs
        # CICIDS2017 contains " Infinity" strings and NaN values
        combined_df = combined_df.replace([np.inf, -np.inf, "Infinity", " Infinity"], np.nan)
        
        missing_val = combined_df.isnull().sum().sum()
        if missing_val > 0:
            print(f"Found {missing_val} missing/infinite values. Dropping rows...")
            combined_df.dropna(inplace=True)

        # 2. Extract Features and Labels
        # The label column is usually named "Label"
        if "Label" not in combined_df.columns:
            raise ValueError("Column 'Label' not found in dataset")

        X = combined_df.drop(["Label"], axis=1)
        y = combined_df["Label"]

        # 3. Ensure all features are numeric
        # Drop any remaining non-numeric columns (e.g. Flow ID, Source IP if present and not dropped)
        non_numeric = X.select_dtypes(exclude=['float32', 'float64', 'int64']).columns
        if len(non_numeric) > 0:
            print(f"Dropping non-numeric feature columns: {non_numeric}")
            X = X.drop(non_numeric, axis=1)

        # 3.1 Drop 'Destination Port' (User Request for Hybrid Model)
        # Note: 'Destination Port' might come as 'Destination Port' or ' Destination Port' (stripped already).
        # We stripped columns at start.
        if 'Destination Port' in X.columns:
            print("Dropping 'Destination Port' feature (User Request)...")
            X = X.drop('Destination Port', axis=1)

        # 4. Encode Labels (Binary Classification)
        # Map "BENIGN" to 0, and all other attacks to 1
        print("Encoding labels for Binary Classification (Benign vs Attack)...")
        y_str = y.astype(str).str.strip()  # Ensure no whitespace issues
        
        # 0 for Benign, 1 for Attack
        y_binary = np.where(y_str == "BENIGN", 0, 1)
        
        # Print stats
        unique, counts = np.unique(y_binary, return_counts=True)
        stats = dict(zip(unique, counts))
        print(f"Label distribution: {stats} (0=Benign, 1=Attack)")
        
        y_encoded = y_binary
        self.num_classes = 2
        print(f"Classes set to {self.num_classes} (Binary)")

        # 5. Normalize Features (MinMax for Hybrid Model Sigmoid Output)
        print("Normalizing Features (MinMaxScaler [0, 1])...")
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scaler.fit_transform(X)
        
        # 5.1 Feature Selection (Added for Fairness/Similarity to STIN-IDS)
        # Select top 25 features to match STIN-IDS model complexity
        # print("Performing Feature Selection (SelectKBest)...")
        # from sklearn.feature_selection import SelectKBest, f_classif
        # # Select top 25 features
        # k = 25
        # selector = SelectKBest(f_classif, k=25)
        # X_scaled = selector.fit_transform(X_scaled, y_encoded)
        # print(f"Reduced feature dimension from {X.shape[1]} to {X_scaled.shape[1]}")

        # Store dimensions
        self.feature_dim = X.shape[1] # Use full dimension for Hybrid Model
        
        # 6. Split Data (Train/Test)
        print("Splitting into Train/Test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
        
        # 7. Convert to Tensors and Store (Required for generate_data)
        self.X_train_tensor = torch.FloatTensor(X_train)
        self.y_train_tensor = torch.LongTensor(y_train)
        self.X_test_tensor = torch.FloatTensor(X_test)
        self.y_test_tensor = torch.LongTensor(y_test)

        print(f"Preprocessing complete.")
        print(f"Train samples: {len(self.X_train_tensor)}, Test samples: {len(self.X_test_tensor)}")
        print(f"Feature Dim: {self.feature_dim}, Num Classes: {self.num_classes}")
        
        # Initializing parent generator fields just in case (though we overwrote attributes)
        # The parent generate_data uses self.X_train_tensor and self.y_train_tensor which we just set.
        
        return self.feature_dim, self.num_classes
