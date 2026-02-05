"""
Interactive Demo: Federated Learning Data Preprocessing
This script demonstrates the complete workflow with explanations
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(step_num, text):
    """Print step information"""
    print(f"\n{'*' * 70}")
    print(f"STEP {step_num}: {text}")
    print('*' * 70)

def main():
    print_header("FEDERATED LEARNING PREPROCESSING - INTERACTIVE DEMO")
    
    print("This demo will walk you through the complete preprocessing pipeline.")
    print("\nWhat you'll learn:")
    print("  1. How to generate a synthetic student performance dataset")
    print("  2. How to preprocess data for federated learning")
    print("  3. How to partition data across multiple clients")
    print("  4. How to visualize the partitioned data")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Generate Dataset
    print_step(1, "Generating Student Performance Dataset")
    print("We'll create a synthetic dataset with 500 students.")
    print("The dataset includes:")
    print("  - Demographics (age, gender)")
    print("  - Academic info (study hours, attendance, grades)")
    print("  - Lifestyle (sleep, exercise)")
    print("  - Socioeconomic factors (family support, internet access)")
    
    input("\nPress Enter to generate dataset...")
    
    import subprocess
    result = subprocess.run([sys.executable, 'generate_dataset.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    if os.path.exists('student_performance.csv'):
        print("✅ Dataset generated successfully!")
    else:
        print("❌ Dataset generation failed!")
        return
    
    input("\nPress Enter to continue to preprocessing...")
    
    # Step 2: Data Preprocessing
    print_step(2, "Data Preprocessing and Partitioning")
    print("Now we'll preprocess the data through these stages:")
    print("  1. Load and explore the dataset")
    print("  2. Handle missing values (mean imputation)")
    print("  3. Encode categorical features")
    print("  4. Normalize numeric features")
    print("  5. Partition data into 5 clients")
    print("  6. Save partitioned data")
    print("  7. Create visualizations")
    
    input("\nPress Enter to start preprocessing...")
    
    # Import and run the preprocessing
    from federated_learning_preprocessing import FederatedDataPreprocessor
    
    print("\nInitializing preprocessor...")
    preprocessor = FederatedDataPreprocessor(
        dataset_path='student_performance.csv',
        num_clients=5
    )
    
    # Execute preprocessing steps
    print("\n" + "-" * 70)
    preprocessor.load_data()
    
    input("\nPress Enter to explore data...")
    print("\n" + "-" * 70)
    preprocessor.explore_data()
    
    input("\nPress Enter to handle missing values...")
    print("\n" + "-" * 70)
    preprocessor.handle_missing_values(strategy='mean')
    
    input("\nPress Enter to encode categorical features...")
    print("\n" + "-" * 70)
    preprocessor.encode_categorical_features()
    
    input("\nPress Enter to normalize features...")
    print("\n" + "-" * 70)
    preprocessor.normalize_features(exclude_target=True, target_column='final_score')
    
    # Partition selection
    print("\n" + "-" * 70)
    print("\nData Partitioning Options:")
    print("  1. IID (Independent and Identically Distributed)")
    print("     - Random distribution")
    print("     - Balanced across clients")
    print("  2. Non-IID (Non-Independent and Identically Distributed)")
    print("     - Sorted distribution")
    print("     - Different data distributions per client")
    
    choice = input("\nChoose partitioning method (1 or 2, default=1): ").strip()
    
    if choice == '2':
        print("\n" + "-" * 70)
        client_data = preprocessor.partition_data_non_iid(partition_column='final_score')
    else:
        print("\n" + "-" * 70)
        client_data = preprocessor.partition_data_iid()
    
    input("\nPress Enter to save partitioned data...")
    print("\n" + "-" * 70)
    output_dir = preprocessor.save_partitioned_data(client_data)
    
    input("\nPress Enter to create visualizations...")
    print("\n" + "-" * 70)
    preprocessor.visualize_partitions(client_data, target_column='final_score')
    
    # Summary
    print_header("PREPROCESSING COMPLETE!")
    
    print("✅ All steps completed successfully!\n")
    print("Generated Files:")
    print("  📁 federated_data/")
    print("     ├── metadata.json")
    for i in range(1, 6):
        print(f"     ├── client_{i}.csv")
    print("     └── partition_visualization.png")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("\n1. Examine the client data files:")
    print("   Each file contains preprocessed data for one client")
    
    print("\n2. View the visualization:")
    print("   Open: federated_data/partition_visualization.png")
    
    print("\n3. Check metadata:")
    print("   Open: federated_data/metadata.json")
    
    print("\n4. Use this data for federated learning training:")
    print("   - Load client data in your FL framework")
    print("   - Train local models on each client")
    print("   - Aggregate model updates")
    print("   - Iterate until convergence")
    
    print("\n" + "=" * 70)
    print("Sample Code to Load Client Data:")
    print("=" * 70)
    print("""
import pandas as pd

# Load client data
client_1_data = pd.read_csv('federated_data/client_1.csv')
client_2_data = pd.read_csv('federated_data/client_2.csv')
# ... and so on

# Separate features and target
X_client_1 = client_1_data.drop('final_score', axis=1)
y_client_1 = client_1_data['final_score']

# Now train your model on client data
# model.fit(X_client_1, y_client_1)
    """)
    
    print("\n" + "=" * 70)
    print("Thank you for using the Federated Learning Preprocessing System!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()