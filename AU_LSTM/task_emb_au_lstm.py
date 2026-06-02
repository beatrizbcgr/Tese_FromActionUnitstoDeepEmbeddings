# ============================================================
# IMPORTS + REPRODUCIBILITY (UNCHANGED)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import random
import pickle

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# LOAD DATA
# ============================================================

stroke_df = pd.read_csv("data/stroke_final.csv")
hc_df     = pd.read_csv("data/hc_final.csv")
df = pd.concat([stroke_df, hc_df], ignore_index=True)

with open("embeddings_and_predictions.pkl", "rb") as f:
    emb_data = pickle.load(f)


def clean_key(k):
    k = k.replace("healthy/", "").replace("stroke/", "")
    k = k.replace("_color", "")
    return k

# CLEAN EMBEDDINGS KEYS
emb_data_clean = {}

for k, v in emb_data.items():
    new_k = clean_key(k)
    emb_data_clean[new_k] = v

emb_data = emb_data_clean

# ============================================================
# MODEL (UNCHANGED)
# ============================================================

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0 ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_norm(x)

        _, (hn, _) = self.lstm(x)

        h = hn[-1]  # last layer hidden state
        h = self.hidden_norm(h)

        return self.fc(h).squeeze(-1)

# ============================================================
# TRAIN FUNCTION (UNCHANGED)
# ============================================================

def train_model(train_loader, input_size):

    model = ImprovedLSTM(input_size=input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(100):

        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)

        train_loss /= len(train_loader.dataset)

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

    model.load_state_dict(best_state)
    return model

# ============================================================
# BUILD SEQUENCES (ADD TASK)
# ============================================================

df["video"] = df["participant_id"].astype(str) + "_" + df["task"].astype(str)
au_cols = [c for c in df.columns if c.startswith("AU")]

video_to_seq = {}
video_to_label = {}
video_to_subject = {}
video_to_task = {}

for v in df["video"].unique():

    sub = df[df["video"] == v].sort_values("frame")

    if v not in emb_data:
        continue

    au_seq = sub[au_cols].values.astype(np.float32)
    emb_seq = np.array(emb_data[v]["embeddings"])

    min_len = min(len(au_seq), len(emb_seq))
    au_seq = au_seq[:min_len]
    emb_seq = emb_seq[:min_len]

    fused_seq = np.concatenate([au_seq, emb_seq], axis=1)

    video_to_seq[v] = fused_seq
    video_to_label[v] = sub["label"].iloc[0]
    video_to_subject[v] = sub["participant_id"].iloc[0]
    video_to_task[v] = sub["task"].iloc[0]

# ============================================================
# CHUNKING (UNCHANGED)
# ============================================================

min_len = min(len(v) for v in video_to_seq.values())

chunks, labels = [], []
video_to_chunks = defaultdict(list)

OVERLAP = 0.30
stride = int(min_len * (1 - OVERLAP))

idx = 0

for v, seq in video_to_seq.items():

    start = 0
    L = len(seq)
    last_start = -1

    while True:

        end = start + min_len

        if end > L:
            end = L
            start = end - min_len

            if start <= last_start:
                break

            chunks.append(seq[start:end])
            labels.append(video_to_label[v])
            video_to_chunks[v].append(idx)
            idx += 1
            break

        chunks.append(seq[start:end])
        labels.append(video_to_label[v])
        video_to_chunks[v].append(idx)
        idx += 1

        last_start = start
        start += stride

X = torch.tensor(np.stack(chunks), dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

subjects = list(set(video_to_subject.values()))
tasks = sorted(set(video_to_task.values()))

# ============================================================
# PER TASK TRAIN + TEST
# ============================================================

from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

print("\n===== RESULTS PER TASK =====")

for task in tasks:

    print(f"\n### TASK: {task} ###")

    task_videos = [v for v in video_to_seq if video_to_task[v] == task]

    all_preds, all_trues = [], []

    for test_subject in subjects:

        train_videos = [v for v in task_videos if video_to_subject[v] != test_subject]
        test_videos  = [v for v in task_videos if video_to_subject[v] == test_subject]

        if len(test_videos) == 0:
            continue

        idx_train = [i for v in train_videos for i in video_to_chunks[v]]

        scaler = StandardScaler().fit(X[idx_train].reshape(-1, X.shape[2]))

        def scale(idx):
            z = scaler.transform(X[idx].reshape(-1, X.shape[2]))
            return torch.tensor(z.reshape(len(idx), min_len, X.shape[2]), dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(scale(idx_train), y[idx_train]),
            batch_size=32,
            shuffle=True
        )

        model = train_model(train_loader, X.shape[2])
        model.eval()

        for v in test_videos:

            idx_test = video_to_chunks[v]
            probs = []

            with torch.no_grad():
                for i_chunk in idx_test:
                    xb = scale([i_chunk])
                    probs.append(torch.sigmoid(model(xb.to(device))).item())

            pred = int(np.mean(probs) > 0.5)
            true = video_to_label[v]

            all_preds.append(pred)
            all_trues.append(true)

    # ========================================================
    # METRICS PER TASK
    # ========================================================

    acc = accuracy_score(all_trues, all_preds)
    f1  = f1_score(all_trues, all_preds)
    sens = recall_score(all_trues, all_preds)
    spec = recall_score(all_trues, all_preds, pos_label=0)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")
    print("Confusion matrix:\n", confusion_matrix(all_trues, all_preds))