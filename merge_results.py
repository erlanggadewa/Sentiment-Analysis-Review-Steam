import glob
import os

import pandas as pd


def merge_all_results():
    """
    Merge all BERT, DistilBERT, and RoBERTa model results into a single CSV file
    """
    # Define the result directory
    result_dir = "result"

    # Pattern to match all model result CSV files (exclude merge-result.csv)
    patterns = ["bert_*.csv", "distilbert_*.csv", "roberta_*.csv"]

    # List to store all dataframes
    all_dataframes = []

    # Process each pattern
    for pattern in patterns:
        file_pattern = os.path.join(result_dir, pattern)
        csv_files = glob.glob(file_pattern)

        for file_path in csv_files:
            # Skip the merge-result.csv file if it exists
            if "merge-result.csv" in file_path:
                continue

            print(f"Processing: {file_path}")

            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Extract learning rate from filename
                filename = os.path.basename(file_path)
                # Extract learning rate (e.g., 1e-05, 2e-05, etc.)
                if "_1e-05.csv" in filename:
                    learning_rate = "1e-05"
                elif "_2e-05.csv" in filename:
                    learning_rate = "2e-05"
                elif "_3e-05.csv" in filename:
                    learning_rate = "3e-05"
                elif "_4e-05.csv" in filename:
                    learning_rate = "4e-05"
                elif "_5e-05.csv" in filename:
                    learning_rate = "5e-05"
                else:
                    learning_rate = "unknown"

                # Add learning rate column
                df["learning_rate"] = learning_rate

                # Extract model type and preprocessing status from filename
                if "distilbert_t_" in filename:
                    model_type = "DistilBERT"
                    preprocessing = "Yes"
                elif "distilbert_nt_" in filename:
                    model_type = "DistilBERT"
                    preprocessing = "No"
                elif "roberta_t_" in filename:
                    model_type = "RoBERTa"
                    preprocessing = "Yes"
                elif "roberta_nt_" in filename:
                    model_type = "RoBERTa"
                    preprocessing = "No"
                elif "bert_t_" in filename:
                    model_type = "BERT"
                    preprocessing = "Yes"
                elif "bert_nt_" in filename:
                    model_type = "BERT"
                    preprocessing = "No"
                else:
                    model_type = "unknown"
                    preprocessing = "unknown"

                # Update model and preprocessing columns to ensure consistency
                df["model"] = model_type
                df["preprocessing"] = preprocessing

                # Add filename for reference
                df["source_file"] = filename

                all_dataframes.append(df)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if all_dataframes:
        # Concatenate all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True)

        # Reorder columns for better readability
        column_order = [
            "model",
            "preprocessing",
            "learning_rate",
            "epoch",
            "train_loss",
            "val_accuracy",
            "val_mcc",
            "val_macro_precision",
            "val_macro_recall",
            "val_macro_f1",
            "test_accuracy",
            "test_mcc",
            "test_macro_precision",
            "test_macro_recall",
            "test_macro_f1",
            "source_file",
        ]

        # Ensure all columns exist before reordering
        existing_columns = [col for col in column_order if col in merged_df.columns]
        merged_df = merged_df[existing_columns]

        # Sort by model, preprocessing, learning_rate, and epoch
        merged_df = merged_df.sort_values(
            ["model", "preprocessing", "learning_rate", "epoch"]
        )

        # Save to merge-result.csv
        output_file = os.path.join(result_dir, "merge-result.csv")
        merged_df.to_csv(output_file, index=False)

        print(f"\nMerged results saved to: {output_file}")
        print(f"Total rows: {len(merged_df)}")
        print(f"Total files processed: {len(all_dataframes)}")

        # Display summary statistics
        print("\nSummary by model:")
        print(merged_df.groupby(["model", "preprocessing"]).size())

        return merged_df
    else:
        print("No CSV files found to merge!")
        return None


if __name__ == "__main__":
    merged_data = merge_all_results()
