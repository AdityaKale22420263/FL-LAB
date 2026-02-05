"""
Federated Learning Data Preprocessing and Partitioning System
Dataset: Student Performance Dataset
Author: FL Assignment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class FederatedDataPreprocessor:
    """
    A class to handle data preprocessing and partitioning for Federated Learning
    """
    
    def __init__(self, dataset_path, num_clients=5):
        """
        Initialize the preprocessor
        
        Parameters:
        -----------
        dataset_path : str
            Path to the student performance dataset CSV file
        num_clients : int
            Number of federated learning clients/nodes
        """
        self.dataset_path = dataset_path
        self.num_clients = num_clients
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the dataset from CSV file"""
        print("=" * 60)
        print("STEP 1: Loading Dataset")
        print("=" * 60)
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Columns: {list(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        return self.df
    
    def explore_data(self):
        """Perform initial data exploration"""
        print("\n" + "=" * 60)
        print("STEP 2: Data Exploration")
        print("=" * 60)
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
        
        return self.df
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Parameters:
        -----------
        strategy : str
            Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        """
        print("\n" + "=" * 60)
        print("STEP 3: Handling Missing Values")
        print("=" * 60)
        
        initial_shape = self.df.shape
        missing_before = self.df.isnull().sum().sum()
        
        print(f"Missing values before: {missing_before}")
        
        if missing_before == 0:
            print("✓ No missing values found. Skipping this step.")
            return self.df
        
        # Separate numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'drop':
            self.df = self.df.dropna()
        
        # Handle categorical columns with mode
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col] = self.df[col].fillna(mode_value)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values after: {missing_after}")
        print(f"✓ Shape: {initial_shape} → {self.df.shape}")
        
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical features to numeric values"""
        print("\n" + "=" * 60)
        print("STEP 4: Encoding Categorical Features")
        print("=" * 60)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            print("✓ No categorical columns found. Skipping encoding.")
            return self.df
        
        print(f"Categorical columns found: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  ✓ Encoded: {col}")
            print(f"    Classes: {list(le.classes_)}")
        
        return self.df
    
    def normalize_features(self, exclude_target=True, target_column=None):
        """
        Normalize features using StandardScaler
        
        Parameters:
        -----------
        exclude_target : bool
            Whether to exclude target column from normalization
        target_column : str
            Name of the target column to exclude
        """
        print("\n" + "=" * 60)
        print("STEP 5: Feature Normalization")
        print("=" * 60)
        
        # Determine columns to normalize
        cols_to_normalize = self.df.columns.tolist()
        
        if exclude_target and target_column and target_column in cols_to_normalize:
            cols_to_normalize.remove(target_column)
            print(f"Excluding target column '{target_column}' from normalization")
        
        print(f"Normalizing {len(cols_to_normalize)} columns...")
        
        # Store original values for comparison
        original_stats = self.df[cols_to_normalize].describe()
        
        # Normalize
        self.df[cols_to_normalize] = self.scaler.fit_transform(self.df[cols_to_normalize])
        
        # Show statistics after normalization
        normalized_stats = self.df[cols_to_normalize].describe()
        
        print("\n✓ Normalization completed!")
        print(f"  Mean before: {original_stats.loc['mean'].mean():.4f}")
        print(f"  Mean after: {normalized_stats.loc['mean'].mean():.4f}")
        print(f"  Std before: {original_stats.loc['std'].mean():.4f}")
        print(f"  Std after: {normalized_stats.loc['std'].mean():.4f}")
        
        return self.df
    
    def partition_data_iid(self):
        """
        Partition data in IID (Independent and Identically Distributed) manner
        Each client gets random samples from the entire dataset
        """
        print("\n" + "=" * 60)
        print("STEP 6: IID Data Partitioning")
        print("=" * 60)
        
        # Shuffle the dataframe
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate samples per client
        total_samples = len(df_shuffled)
        samples_per_client = total_samples // self.num_clients
        
        print(f"Total samples: {total_samples}")
        print(f"Samples per client: {samples_per_client}")
        
        client_data = {}
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            
            # Last client gets remaining samples
            if i == self.num_clients - 1:
                end_idx = total_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_data[f'client_{i+1}'] = df_shuffled.iloc[start_idx:end_idx].reset_index(drop=True)
            print(f"  ✓ Client {i+1}: {len(client_data[f'client_{i+1}'])} samples")
        
        return client_data
    
    def partition_data_non_iid(self, partition_column=None):
        """
        Partition data in Non-IID manner based on a specific column
        Each client gets data with different distributions
        
        Parameters:
        -----------
        partition_column : str
            Column name to use for non-IID partitioning
        """
        print("\n" + "=" * 60)
        print("STEP 6: Non-IID Data Partitioning")
        print("=" * 60)
        
        if partition_column is None:
            # Use the last column as partition column
            partition_column = self.df.columns[-1]
        
        print(f"Partitioning based on column: {partition_column}")
        
        # Sort by partition column
        df_sorted = self.df.sort_values(by=partition_column).reset_index(drop=True)
        
        # Calculate samples per client
        total_samples = len(df_sorted)
        samples_per_client = total_samples // self.num_clients
        
        print(f"Total samples: {total_samples}")
        print(f"Samples per client: ~{samples_per_client}")
        
        client_data = {}
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            
            # Last client gets remaining samples
            if i == self.num_clients - 1:
                end_idx = total_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_data[f'client_{i+1}'] = df_sorted.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # Show distribution statistics
            client_df = client_data[f'client_{i+1}']
            print(f"  ✓ Client {i+1}: {len(client_df)} samples")
            print(f"    {partition_column} range: [{client_df[partition_column].min():.2f}, {client_df[partition_column].max():.2f}]")
        
        return client_data
    
    def save_partitioned_data(self, client_data, output_dir='federated_data'):
        """
        Save partitioned data to separate CSV files
        
        Parameters:
        -----------
        client_data : dict
            Dictionary containing client dataframes
        output_dir : str
            Directory to save client data files
        """
        print("\n" + "=" * 60)
        print("STEP 7: Saving Partitioned Data")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'num_clients': self.num_clients,
            'total_samples': len(self.df),
            'num_features': len(self.df.columns),
            'feature_names': list(self.df.columns),
            'client_samples': {k: len(v) for k, v in client_data.items()}
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✓ Metadata saved to {output_dir}/metadata.json")
        
        # Save each client's data
        for client_name, client_df in client_data.items():
            filepath = f'{output_dir}/{client_name}.csv'
            client_df.to_csv(filepath, index=False)
            print(f"✓ {client_name}: Saved {len(client_df)} samples to {filepath}")
        
        print(f"\n✓ All data saved to '{output_dir}/' directory")
        return output_dir
    
    def visualize_partitions(self, client_data, target_column=None):
        """
        Create visualizations for the partitioned data
        
        Parameters:
        -----------
        client_data : dict
            Dictionary containing client dataframes
        target_column : str
            Target column for distribution visualization
        """
        print("\n" + "=" * 60)
        print("STEP 8: Visualizing Data Partitions")
        print("=" * 60)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Federated Learning Data Partitions Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution of samples across clients
        ax1 = axes[0, 0]
        client_names = list(client_data.keys())
        sample_counts = [len(client_data[name]) for name in client_names]
        
        bars = ax1.bar(client_names, sample_counts, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Clients', fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontweight='bold')
        ax1.set_title('Sample Distribution Across Clients')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Target distribution (if target column is provided)
        ax2 = axes[0, 1]
        if target_column and target_column in self.df.columns:
            for client_name in client_names:
                client_df = client_data[client_name]
                ax2.hist(client_df[target_column], alpha=0.5, label=client_name, bins=20)
            
            ax2.set_xlabel(target_column, fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title(f'{target_column} Distribution Across Clients')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No target column specified', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Target Distribution')
        
        # 3. Feature correlation heatmap (using first client as example)
        ax3 = axes[1, 0]
        first_client_df = client_data[client_names[0]]
        correlation_matrix = first_client_df.corr()
        
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(correlation_matrix.columns)))
        ax3.set_yticks(range(len(correlation_matrix.columns)))
        ax3.set_xticklabels(correlation_matrix.columns, rotation=90, ha='right', fontsize=8)
        ax3.set_yticklabels(correlation_matrix.columns, fontsize=8)
        ax3.set_title(f'Feature Correlation Heatmap\n(Client 1 Example)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        # 4. Data statistics comparison
        ax4 = axes[1, 1]
        stats_data = []
        for client_name in client_names:
            client_df = client_data[client_name]
            stats_data.append({
                'Client': client_name,
                'Mean': client_df.mean().mean(),
                'Std': client_df.std().mean(),
                'Min': client_df.min().min(),
                'Max': client_df.max().max()
            })
        
        stats_df = pd.DataFrame(stats_data)
        x = np.arange(len(client_names))
        width = 0.2
        
        ax4.bar(x - width*1.5, stats_df['Mean'], width, label='Mean', alpha=0.8)
        ax4.bar(x - width*0.5, stats_df['Std'], width, label='Std', alpha=0.8)
        ax4.bar(x + width*0.5, stats_df['Min'], width, label='Min', alpha=0.8)
        ax4.bar(x + width*1.5, stats_df['Max'], width, label='Max', alpha=0.8)
        
        ax4.set_xlabel('Clients', fontweight='bold')
        ax4.set_ylabel('Values', fontweight='bold')
        ax4.set_title('Statistical Comparison Across Clients')
        ax4.set_xticks(x)
        ax4.set_xticklabels(client_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = 'federated_data/partition_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {viz_path}")
        
        plt.close()
        
        return viz_path


def main():
    """
    Main function to demonstrate the federated learning preprocessing pipeline
    """
    print("=" * 60)
    print("FEDERATED LEARNING DATA PREPROCESSING SYSTEM")
    print("=" * 60)
    print()
    
    # Configuration
    DATASET_PATH = 'student_performance.csv'  # Make sure this file exists
    NUM_CLIENTS = 5
    PARTITION_TYPE = 'iid'  # Options: 'iid' or 'non_iid'
    TARGET_COLUMN = 'final_score'  # Adjust based on your dataset
    
    try:
        # Initialize preprocessor
        preprocessor = FederatedDataPreprocessor(
            dataset_path=DATASET_PATH,
            num_clients=NUM_CLIENTS
        )
        
        # Step 1: Load data
        preprocessor.load_data()
        
        # Step 2: Explore data
        preprocessor.explore_data()
        
        # Step 3: Handle missing values
        preprocessor.handle_missing_values(strategy='mean')
        
        # Step 4: Encode categorical features
        preprocessor.encode_categorical_features()
        
        # Step 5: Normalize features
        preprocessor.normalize_features(
            exclude_target=True,
            target_column=TARGET_COLUMN
        )
        
        # Step 6: Partition data
        if PARTITION_TYPE == 'iid':
            client_data = preprocessor.partition_data_iid()
        else:
            client_data = preprocessor.partition_data_non_iid(
                partition_column=TARGET_COLUMN
            )
        
        # Step 7: Save partitioned data
        output_dir = preprocessor.save_partitioned_data(client_data)
        
        # Step 8: Visualize partitions
        preprocessor.visualize_partitions(
            client_data,
            target_column=TARGET_COLUMN
        )
        
        print("\n" + "=" * 60)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nData partitioned into {NUM_CLIENTS} clients")
        print(f"Output directory: {output_dir}/")
        print("\nFiles created:")
        print("  - metadata.json")
        for i in range(1, NUM_CLIENTS + 1):
            print(f"  - client_{i}.csv")
        print("  - partition_visualization.png")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Dataset file '{DATASET_PATH}' not found!")
        print("Please make sure the dataset is in the same directory as this script.")
        print("\nExpected filename: student_performance.csv")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()