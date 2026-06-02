import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
import random

# ============================================================
# LOAD DATA
# ============================================================

stroke_df = pd.read_csv("data/stroke_final.csv")
hc_df     = pd.read_csv("data/hc_final.csv")

df = pd.concat([stroke_df, hc_df], ignore_index=True)

# ============================================================
# REPRODUCIBILITY + DEVICE
# ============================================================

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
# TRAINING SETTINGS
# ============================================================

MAX_EPOCHS = 100
PATIENCE   = 40

# ============================================================
# LSTM MODEL (NO ATTENTION)
# ============================================================

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_norm(x)
        _, (hn, _) = self.lstm(x)
        h = hn[-1]
        h = self.hidden_norm(h)
        return self.fc(h).squeeze(-1)

# ============================================================
# TRAINING (NO VALIDATION, WITH TEST CURVE LOGGING)
# ============================================================

def train_model(train_loader, input_size, log_dir,
                scale, test_video, video_to_chunks, video_to_label,
                X, min_len):

    model = ImprovedLSTM(input_size=input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-5
    )

    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        # ---------------- TRAIN ----------------
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

        # ---------------- TEST ACC (VIDEO LEVEL) ----------------
        model.eval()
        probs = []

        with torch.no_grad():
            for idx in video_to_chunks[test_video]:
                xb = scale([idx])
                probs.append(torch.sigmoid(model(xb.to(device))).item())

        pred = int(np.mean(probs) > 0.5)
        true = video_to_label[test_video]
        test_acc = int(pred == true)

        # ---------------- LOGGING ----------------
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        #print(
        #    f"Epoch {epoch:03d} | "
        #    f"TrainLoss={train_loss:.4f} | "
        #    f"TestAcc={test_acc:.4f}"
        #)

        # ---------------- MODEL SELECTION ----------------
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    writer.close()

    model.load_state_dict(best_state)
    return model

# ============================================================
# DATA PREPARATION
# ============================================================

df["video"] = df["participant_id"].astype(str) + "_" + df["task"].astype(str)

# ONLY THESE 3 AUs
au_cols = ["AU23", "AU07", "AU20"]

video_to_seq, video_to_label, video_to_task = {}, {}, {}

for v in df["video"].unique():
    sub = df[df["video"] == v]
    video_to_seq[v] = sub[au_cols].values.astype(np.float32)
    video_to_label[v] = sub["label"].iloc[0]
    video_to_task[v] = sub["task"].iloc[0]

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

# ============================================================
# LOSO PER TASK (NO VALIDATION)
# ============================================================

for task in df["task"].unique():

    print(f"\n=== TASK: {task} ===")
    videos_task = [v for v in video_to_seq if video_to_task[v] == task]
    if len(videos_task) < 3:
        continue

    task_preds = []
    task_trues = []

    for test_video in videos_task:

        train_videos = [v for v in videos_task if v != test_video]

        idx_train = [i for v in train_videos for i in video_to_chunks[v]]
        idx_test  = video_to_chunks[test_video]

        scaler = StandardScaler().fit(X[idx_train].reshape(-1, X.shape[2]))

        def scale(idx):
            z = scaler.transform(X[idx].reshape(-1, X.shape[2]))
            return torch.tensor(
                z.reshape(len(idx), min_len, X.shape[2]),
                dtype=torch.float32
            )

        train_loader = DataLoader(
            TensorDataset(scale(idx_train), y[idx_train]),
            batch_size=8,
            shuffle=True
        )

        log_dir = f"runs/au_lstm_testcurve_overlap_3aus/{task}/{test_video}"
        model = train_model(
            train_loader,
            len(au_cols),
            log_dir,
            scale,
            test_video,
            video_to_chunks,
            video_to_label,
            X,
            min_len
        )

        # -------- FINAL TEST (VIDEO LEVEL) --------
        model.eval()
        probs = []

        with torch.no_grad():
            for idx_chunk in idx_test:
                xb = scale([idx_chunk])
                probs.append(torch.sigmoid(model(xb.to(device))).item())

        pred = int(np.mean(probs) > 0.5)
        true = video_to_label[test_video]

        task_preds.append(pred)
        task_trues.append(true)

    if len(task_trues) > 0:
        task_acc = np.mean(np.array(task_preds) == np.array(task_trues))
        print(
            f">>> TASK {task} | "
            f"Video-ACC = {task_acc:.3f} "
            f"({len(task_trues)} videos)"
        )