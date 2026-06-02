import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
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
PATIENCE   = 50

# ============================================================
# SIMPLE RNN (ONLY CHANGE HERE)
# ============================================================

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()

        self.norm = nn.LayerNorm(input_size)

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True
        )

        self.hnorm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.norm(x)

        _, hn = self.rnn(x)   # <-- DIFFERENT FROM LSTM

        h = hn[-1]            # last layer hidden state
        h = self.hnorm(h)

        return self.fc(h).squeeze(-1)

# ============================================================
# TRAINING
# ============================================================

def train_model(train_loader, input_size, log_dir,
                scale, test_videos, video_to_chunks, video_to_label):

    model = SimpleRNN(input_size=input_size).to(device)  # <-- CHANGE

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

        # ---------------- TEST ACC ----------------
        model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for v in test_videos:

                chunk_probs = []

                for idx in video_to_chunks[v]:
                    xb = scale([idx])
                    p = torch.sigmoid(model(xb.to(device))).item()
                    chunk_probs.append(p)

                pred = int(np.mean(chunk_probs) > 0.5)

                preds.append(pred)
                trues.append(video_to_label[v])

        test_acc = np.mean(np.array(preds) == np.array(trues))

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        print(f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | TestAcc={test_acc:.4f}")

        # EARLY STOPPING
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
# DATA PREPARATION (UNCHANGED)
# ============================================================

df["video"] = df["participant_id"].astype(str) + "_" + df["task"].astype(str)

au_cols = [c for c in df.columns if c.startswith("AU")]

video_to_seq, video_to_label, video_to_subject = {}, {}, {}

for v in df["video"].unique():

    sub = df[df["video"] == v].sort_values("frame")

    video_to_seq[v] = sub[au_cols].values.astype(np.float32)
    video_to_label[v] = sub["label"].iloc[0]
    video_to_subject[v] = sub["participant_id"].iloc[0]

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

subjects = df["participant_id"].unique()

# ============================================================
# GLOBAL LOSO
# ============================================================

all_preds = []
all_trues = []

for test_subject in subjects:

    print(f"\n=== TEST SUBJECT: {test_subject} ===")

    train_videos = [v for v in video_to_seq if video_to_subject[v] != test_subject]
    test_videos  = [v for v in video_to_seq if video_to_subject[v] == test_subject]

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

    log_dir = f"runs/global_au_rnn_testcurve_overlap/{test_subject}"

    model = train_model(
        train_loader,
        len(au_cols),
        log_dir,
        scale,
        test_videos,
        video_to_chunks,
        video_to_label
    )

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

# ============================================================
# FINAL RESULTS
# ============================================================

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

acc = np.mean(all_preds == all_trues)

print("\n=== FINAL GLOBAL RESULTS (RNN) ===")
print(f"Video-Accuracy: {acc:.3f}")