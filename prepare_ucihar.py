import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).parent
# Unzip the UCI HAR download so that the 'UCI HAR Dataset' folder sits next to this script
DATA_DIR = BASE_DIR / 'UCI HAR Dataset'
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}\nDownload it from https://archive.ics.uci.edu/dataset/240")

# ── Load feature names and make unique ──
features = pd.read_csv(DATA_DIR / 'features.txt', sep=r'\s+', header=None, names=['id','name'])
feat_names = features['name'].tolist()

seen = {}
unique_names = []
for name in feat_names:
    if name in seen:
        seen[name] += 1
        unique_names.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 0
        unique_names.append(name)

# ── Activity labels ──
activity_map = {1:'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS',
                4:'SITTING', 5:'STANDING', 6:'LAYING'}

# ── Load official train split ──
X_train = pd.read_csv(DATA_DIR / 'train' / 'X_train.txt', sep=r'\s+', header=None, names=unique_names)
y_train = pd.read_csv(DATA_DIR / 'train' / 'y_train.txt', header=None, names=['activity'])
s_train = pd.read_csv(DATA_DIR / 'train' / 'subject_train.txt', header=None, names=['subject'])
y_train['activity'] = y_train['activity'].map(activity_map)

# ── Load official test split ──
X_test = pd.read_csv(DATA_DIR / 'test' / 'X_test.txt', sep=r'\s+', header=None, names=unique_names)
y_test = pd.read_csv(DATA_DIR / 'test' / 'y_test.txt', header=None, names=['activity'])
s_test = pd.read_csv(DATA_DIR / 'test' / 'subject_test.txt', header=None, names=['subject'])
y_test['activity'] = y_test['activity'].map(activity_map)

# ── Combine features + labels + subject ──
train_df = pd.concat([X_train, y_train, s_train], axis=1)
test_df  = pd.concat([X_test,  y_test,  s_test],  axis=1)

train_df['split'] = 'train'
test_df['split']  = 'test'

df = pd.concat([train_df, test_df]).reset_index(drop=True)

print("Shape:", df.shape)
print("\nActivity distribution:")
print(df['activity'].value_counts())
print("\nSubjects in train:", sorted(s_train['subject'].unique()))
print("Subjects in test: ", sorted(s_test['subject'].unique()))
print("\nTrain size:", len(train_df))
print("Test size: ", len(test_df))

# ── Save ──
df.to_csv(BASE_DIR / 'ucihar_combined.csv', index=False)
train_df.to_csv(BASE_DIR / 'ucihar_train.csv', index=False)
test_df.to_csv(BASE_DIR / 'ucihar_test.csv',  index=False)

print("\n✅ Saved ucihar_combined.csv")
print("✅ Saved ucihar_train.csv")
print("✅ Saved ucihar_test.csv")