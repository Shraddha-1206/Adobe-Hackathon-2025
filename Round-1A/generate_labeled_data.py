import os
import pandas as pd
from scripts.extract_features import extract_lines_with_features  # Existing function

input_folder = "input"  # Folder jisme aapke PDFs hai
all_data = []

# Sab PDF files loop me process karo
for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        print(f" Processing: {filename}")
        pdf_path = os.path.join(input_folder, filename)
        df = extract_lines_with_features(pdf_path)  # Existing function call
        df["label"] = "body"  # Sabko default label do, baad me aap manually label karoge
        df["source_file"] = filename  # Track karne ke liye kaunse file se line aayi
        all_data.append(df)

# Combine sabhi PDFs ka data ek saath
combined_df = pd.concat(all_data, ignore_index=True)

# Save as labeled_data.csv
combined_df.to_csv("labeled_data.csv", index=False)
print(" Done! labeled_data.csv file ban gayi.")
