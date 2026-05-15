import os
import random

import pandas as pd

SEED: int = 42

random.seed(SEED)

real_dir = os.path.join("data", "real")
fake_dir = os.path.join("data", "fake")

len_real = len(os.listdir(real_dir))
len_fake = len(os.listdir(fake_dir))

# Split the data into 70-20-10 for train, test, and validation
train_real = int(len_real * 0.7)
val_real = int(len_real * 0.2)
test_real = len_real - train_real - val_real

train_fake = int(len_fake * 0.7)
val_fake = int(len_fake * 0.2)
test_fake = len_fake - train_fake - val_fake

real_files = os.listdir(real_dir)
fake_files = os.listdir(fake_dir)

random.shuffle(real_files)
random.shuffle(fake_files)

# Slice into splits
train_real_files = real_files[:train_real]
val_real_files = real_files[train_real : train_real + val_real]
test_real_files = real_files[train_real + val_real :]

train_fake_files = fake_files[:train_fake]
val_fake_files = fake_files[train_fake : train_fake + val_fake]
test_fake_files = fake_files[train_fake + val_fake :]

# Build DataFrames
train_data = [(os.path.join(real_dir, f), 1) for f in train_real_files] + [
    (os.path.join(fake_dir, f), 0) for f in train_fake_files
]
val_data = [(os.path.join(real_dir, f), 1) for f in val_real_files] + [
    (os.path.join(fake_dir, f), 0) for f in val_fake_files
]
test_data = [(os.path.join(real_dir, f), 1) for f in test_real_files] + [
    (os.path.join(fake_dir, f), 0) for f in test_fake_files
]

train_df = pd.DataFrame(train_data, columns=["file_path", "label"])
val_df = pd.DataFrame(val_data, columns=["file_path", "label"])
test_df = pd.DataFrame(test_data, columns=["file_path", "label"])

# Save to CSV
train_df.to_csv(os.path.join("data", "train.csv"), index=False)
val_df.to_csv(os.path.join("data", "val.csv"), index=False)
test_df.to_csv(os.path.join("data", "test.csv"), index=False)
