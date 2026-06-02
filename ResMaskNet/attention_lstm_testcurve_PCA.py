# ============================================================
# REPRODUCIBILITY
# ============================================================
import random
SEED = 42
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# IMPORTS
# ============================================================
import pickle
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict

# ============================================================
# ATTENTION MODEL
# ============================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * lstm_out, dim=1)


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True
        )
        self.attention = TemporalAttention(hidden_size)
        self.hnorm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        h = self.attention(lstm_out)
        h = self.hnorm(h)
        return self.fc(h).squeeze(-1)

# ============================================================
# LOAD DATA
# ============================================================
with open("embeddings_and_predictions.pkl", "rb") as f:
    data = pickle.load(f)

all_lengths = [len(info["embeddings"]) for info in data.values()]
min_len = min(all_lengths)
print("Min sequence length:", min_len)

def extract_task(video):
    parts = video.split("/")[-1].split("_")
    return parts[2] + "_" + parts[3]

def extract_subject(video):
    return video.split("/")[-1].split("_")[0]

chunks, labels = [], []
video_to_chunks, video_to_label, video_to_task = {}, {}, {}
subject_to_videos = defaultdict(list)

i = 0

SKIP_TASKS = {"NSM_BIGSMILE", "NSM_BROW"}
OVERLAP = 0.30
stride = int(min_len * (1 - OVERLAP))

for video, info in data.items():
    emb = np.array(info["embeddings"])
    label = 1 if "stroke" in video.lower() else 0
    task = extract_task(video)
    subject = extract_subject(video)

    if task in SKIP_TASKS:
        continue

    start = 0
    L = len(emb)
    last_start = -1

    while True:
        end = start + min_len

        if end > L:
            end = L
            start = end - min_len

            if start <= last_start:
                break

            chunks.append(torch.tensor(emb[start:end], dtype=torch.float32))
            labels.append(label)
            video_to_chunks.setdefault(video, []).append(i)
            i += 1
            break

        chunks.append(torch.tensor(emb[start:end], dtype=torch.float32))
        labels.append(label)
        video_to_chunks.setdefault(video, []).append(i)
        i += 1

        last_start = start
        start += stride

    video_to_label[video] = label
    video_to_task[video] = task
    subject_to_videos[subject].append(video)

X = torch.stack(chunks)
y = torch.tensor(labels, dtype=torch.float32)

subjects = list(subject_to_videos.keys())

# ============================================================
# TRAIN FUNCTION
# ============================================================
def train_one(train_loader, scale, test_videos, subject_id, n_comp,
              max_epochs=100, patience=40):

    model = AttentionLSTM(input_size=n_comp).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-5
    )

    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir=f"runs_globalattention111/pca111_{n_comp}/subject_{subject_id}")

    best_train_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):

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

        # ---------------- TEST ----------------
        model.eval()
        test_preds, test_trues = [], []

        with torch.no_grad():
            for v in test_videos:
                chunk_preds = []

                for idx in video_to_chunks[v]:
                    xb = scale(idx).unsqueeze(0).to(device)
                    p = torch.sigmoid(model(xb)).item()
                    chunk_preds.append(p > 0.5)

                pred_video = int(sum(chunk_preds) > len(chunk_preds) / 2)
                test_preds.append(pred_video)
                test_trues.append(video_to_label[v])

        test_acc = accuracy_score(test_trues, test_preds)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        print(f"[PCA {n_comp} | {subject_id}] Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | TestAcc={test_acc:.4f}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    writer.close()
    model.load_state_dict(best_state)
    return model

# ============================================================
# PCA EXPERIMENT
# ============================================================
pca_dims = [8, 16]
results = {}

# NEW: STORE VIDEO PREDICTIONS
all_prediction_rows = []

for n_comp in pca_dims:

    print(f"\n===== PCA DIM: {n_comp} =====")

    all_video_preds, all_video_trues = [], []

    for test_subject in subjects:

        train_subjects = [s for s in subjects if s != test_subject]

        idx_train = []
        for s in train_subjects:
            for v in subject_to_videos[s]:
                idx_train.extend(video_to_chunks[v])

        test_videos = subject_to_videos[test_subject]

        # ---------------- STANDARDIZATION ----------------
        scaler = StandardScaler().fit(X[idx_train].reshape(-1, X.shape[2]))

        X_scaled = scaler.transform(X.reshape(-1, X.shape[2]))
        X_scaled = X_scaled.reshape(X.shape)

        # ---------------- PCA ----------------
        pca = PCA(n_components=n_comp)

        X_train_2d = X_scaled[idx_train].reshape(-1, X.shape[2])
        pca.fit(X_train_2d)

        X_pca_2d = pca.transform(X_scaled.reshape(-1, X.shape[2]))
        X_pca = X_pca_2d.reshape(X.shape[0], X.shape[1], n_comp)

        def scale(idx):
            return torch.tensor(X_pca[idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(scale(idx_train), y[idx_train]),
            batch_size=32,
            shuffle=True
        )

        model = train_one(
            train_loader,
            scale,
            test_videos,
            test_subject,
            n_comp
        )

        model.eval()

        for v in test_videos:
            chunk_preds = []
            chunk_probs = []

            with torch.no_grad():
                for idx in video_to_chunks[v]:
                    xb = scale(idx).unsqueeze(0).to(device)
                    p = torch.sigmoid(model(xb)).item()

                    chunk_probs.append(float(p))
                    chunk_preds.append(int(p > 0.5))

            pred_video = int(sum(chunk_preds) > len(chunk_preds) / 2)
            true_video = video_to_label[v]
            proba_video = float(np.mean(chunk_probs))

            all_video_preds.append(pred_video)
            all_video_trues.append(true_video)

            # SAVE VIDEO RESULTS
            all_prediction_rows.append({
                "pca_dimensions": n_comp,
                "video_name": v.split("/")[-1],
                "video_path": v,
                "subject": test_subject,
                "task": video_to_task[v],
                "true_label": true_video,
                "predicted_label": pred_video,
                "predicted_probability": proba_video,
                "correct": int(pred_video == true_video)
            })

    acc = accuracy_score(all_video_trues, all_video_preds)
    f1  = f1_score(all_video_trues, all_video_preds)

    results[n_comp] = (acc, f1)

    print(f"RESULTS → Acc={acc:.3f} | F1={f1:.3f}")

# ============================================================
# SAVE VIDEO PREDICTIONS TO EXCEL
# ============================================================
predictions_df = pd.DataFrame(all_prediction_rows)
predictions_df.to_excel("attention_lstm_pca_global_video_predictions.xlsx", index=False)

print("Saved predictions to: attention_lstm_pca_global_video_predictions.xlsx")

# ============================================================
# FINAL
# ============================================================
print("\n===== FINAL COMPARISON =====")
for n, (acc, f1) in results.items():
    print(f"PCA {n:>2} dims -> Acc: {acc:.3f} | F1: {f1:.3f}")