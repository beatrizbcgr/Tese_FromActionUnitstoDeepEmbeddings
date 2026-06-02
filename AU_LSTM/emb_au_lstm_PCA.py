# ============================================================
# IMPORTS + REPRODUCIBILITY (UNCHANGED)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
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
# LOAD DATA (UNCHANGED)
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

emb_data = {clean_key(k): v for k, v in emb_data.items()}

# ============================================================
# MODEL (UNCHANGED)
# ============================================================

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=0.3, batch_first=True)
        self.hnorm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.norm(x)
        _, (hn, _) = self.lstm(x)
        h = self.hnorm(hn[-1])
        return self.fc(h).squeeze(-1)

# ============================================================
# BUILD SEQUENCES (SEPARATE AU + EMB)
# ============================================================

df["video"] = df["participant_id"].astype(str) + "_" + df["task"].astype(str)
au_cols = [c for c in df.columns if c.startswith("AU")]

video_to_au = {}
video_to_emb = {}
video_to_label = {}
video_to_subject = {}

for v in df["video"].unique():

    sub = df[df["video"] == v].sort_values("frame")

    if v not in emb_data:
        continue

    au_seq  = sub[au_cols].values.astype(np.float32)
    emb_seq = np.array(emb_data[v]["embeddings"])

    min_len = min(len(au_seq), len(emb_seq))
    video_to_au[v]  = au_seq[:min_len]
    video_to_emb[v] = emb_seq[:min_len]

    video_to_label[v] = sub["label"].iloc[0]
    video_to_subject[v] = sub["participant_id"].iloc[0]

subjects = sorted(set(video_to_subject.values()))

# ============================================================
# CHUNKING (UNCHANGED LOGIC)
# ============================================================

min_len = min(len(v) for v in video_to_au.values())

chunks_au, chunks_emb, labels = [], [], []
video_to_chunks = defaultdict(list)

OVERLAP = 0.30
stride = int(min_len * (1 - OVERLAP))

idx = 0

for v in video_to_au:

    au_seq  = video_to_au[v]
    emb_seq = video_to_emb[v]

    start = 0
    L = len(au_seq)
    last_start = -1

    while True:

        end = start + min_len

        if end > L:
            end = L
            start = end - min_len
            if start <= last_start:
                break

            chunks_au.append(au_seq[start:end])
            chunks_emb.append(emb_seq[start:end])
            labels.append(video_to_label[v])
            video_to_chunks[v].append(idx)
            idx += 1
            break

        chunks_au.append(au_seq[start:end])
        chunks_emb.append(emb_seq[start:end])
        labels.append(video_to_label[v])
        video_to_chunks[v].append(idx)
        idx += 1

        last_start = start
        start += stride

X_au  = np.stack(chunks_au)
X_emb = np.stack(chunks_emb)
y = torch.tensor(labels, dtype=torch.float32)

subjects = sorted(set(video_to_subject.values()))

# ============================================================
# TRAIN FUNCTION (UNCHANGED)
# ============================================================

def train_model(train_loader, input_size):
    model = SimpleLSTM(input_size=input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-5)

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
        scheduler.step(epoch)

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
# PCA EXPERIMENT (ONLY ON EMBEDDINGS)
# ============================================================

pca_dims = [8, 16]
results = {}

for n_comp in pca_dims:

    print(f"\n===== PCA DIM: {n_comp} =====")

    all_preds, all_trues = [], []

    for test_subject in subjects:

        train_videos = [v for v in video_to_au if video_to_subject[v] != test_subject]
        test_videos  = [v for v in video_to_au if video_to_subject[v] == test_subject]

        idx_train = [i for v in train_videos for i in video_to_chunks[v]]

        # -------- STANDARDIZE EMB ONLY --------
        scaler = StandardScaler().fit(X_emb[idx_train].reshape(-1, X_emb.shape[2]))

        X_emb_scaled = scaler.transform(X_emb.reshape(-1, X_emb.shape[2]))
        X_emb_scaled = X_emb_scaled.reshape(X_emb.shape)

        # -------- PCA ON EMB --------
        pca = PCA(n_components=n_comp)
        pca.fit(X_emb_scaled[idx_train].reshape(-1, X_emb.shape[2]))

        X_emb_pca = pca.transform(X_emb_scaled.reshape(-1, X_emb.shape[2]))
        X_emb_pca = X_emb_pca.reshape(X_emb.shape[0], X_emb.shape[1], n_comp)

        # -------- CONCAT BACK --------
        X_fused = np.concatenate([X_au, X_emb_pca], axis=2)

        def scale(idx):
            return torch.tensor(X_fused[idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(scale(idx_train), y[idx_train]),
            batch_size=32,
            shuffle=True
        )

        writer = SummaryWriter(f"runs/au_emb_pca_{n_comp}/{test_subject}")

        model = train_model(train_loader, X_fused.shape[2])

        model.eval()

        for v in test_videos:
            probs = []

            with torch.no_grad():
                for i_chunk in video_to_chunks[v]:
                    xb = scale([i_chunk]).to(device)
                    probs.append(torch.sigmoid(model(xb)).item())

            pred = int(np.mean(probs) > 0.5)
            all_preds.append(pred)
            all_trues.append(video_to_label[v])

        writer.close()

    acc = accuracy_score(all_trues, all_preds)
    f1  = f1_score(all_trues, all_preds)

    results[n_comp] = (acc, f1)

    print(f"RESULTS → Acc={acc:.3f} | F1={f1:.3f}")

# ============================================================
# FINAL
# ============================================================

print("\n===== FINAL COMPARISON =====")
for n, (acc, f1) in results.items():
    print(f"PCA {n:>2} dims -> Acc: {acc:.3f} | F1: {f1:.3f}")