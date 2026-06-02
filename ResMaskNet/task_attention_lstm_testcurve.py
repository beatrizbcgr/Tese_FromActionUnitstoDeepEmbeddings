# ============================================================
# REPRODUCIBILITY + DEVICE
# ============================================================
import random
import numpy as np
import torch

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
# IMPORTS
# ============================================================
import pickle
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict
import os

# ============================================================
# TEMPORAL ATTENTION
# ============================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

# ============================================================
# LSTM + ATTENTION MODEL (UNCHANGED)
# ============================================================
class AttentionLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
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
stride = int(min_len * (1 - OVERLAP))   # 70% stride

for video, info in data.items():
    emb = np.array(info["embeddings"])
    label = 1 if "stroke" in video.lower() else 0
    task = extract_task(video)
    subject = extract_subject(video)

    if task in SKIP_TASKS:
        continue

    start = 0
    L = len(emb)
    last_start = -1   # prevent duplicates

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
# GROUP BY TASK
# ============================================================
task_to_videos = defaultdict(list)
for v in video_to_chunks:
    task_to_videos[video_to_task[v]].append(v)

# ============================================================
# MAIN LOOP (LOSO PER TASK)
# ============================================================
all_preds, all_trues = [], []

# NEW: STORE VIDEO PREDICTIONS
fold_results = []

for task, videos in task_to_videos.items():

    if task in {"NSM_BIGSMILE", "NSM_BROW"}:
        continue

    print(f"\n=== TASK: {task} ===")
    task_preds, task_trues = [], []

    for test_video in videos:

        train_idx = [i for v in videos if v != test_video for i in video_to_chunks[v]]
        test_idx = video_to_chunks[test_video]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]

        # -------------------------
        # STANDARDIZATION (TRAIN ONLY)
        # -------------------------
        scaler = StandardScaler().fit(
            X_train.reshape(-1, X.shape[2])
        )

        X_train = torch.tensor(
            scaler.transform(X_train.reshape(-1, X.shape[2]))
            .reshape(X_train.shape),
            dtype=torch.float32
        )
        X_test = torch.tensor(
            scaler.transform(X_test.reshape(-1, X.shape[2]))
            .reshape(X_test.shape),
            dtype=torch.float32
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=8,
            shuffle=True
        )

        model = AttentionLSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=1, eta_min=1e-5
        )

        criterion = nn.BCEWithLogitsLoss()

        writer = SummaryWriter(
            log_dir=f"runs/task_attention_lstm_testcurve_overlap/{task}/{test_video.replace('/', '_')}"
        )

        best_train = float("inf")
        best_state = None
        counter = 0
        patience = 40

        # ============================================================
        # TRAIN
        # ============================================================
        for epoch in range(100):

            model.train()
            total_loss = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * yb.size(0)

            train_loss = total_loss / len(train_loader.dataset)

            scheduler.step(epoch)

            # ====================================================
            # TEST ACCURACY (VIDEO LEVEL, ONLY FOR CURVE)
            # ====================================================
            model.eval()
            probs = []
            with torch.no_grad():
                for xb in X_test:
                    probs.append(torch.sigmoid(model(xb.unsqueeze(0).to(device))).item())

            pred_epoch = int(np.mean(probs) > 0.5)
            true_epoch = video_to_label[test_video]
            test_acc = int(pred_epoch == true_epoch)

            # ====================================================
            # LOGGING
            # ====================================================
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            print(
                f"Epoch {epoch+1:03d} | "
                f"TrainLoss={train_loss:.4f} | "
                f"TestAcc={test_acc}"
            )

            # ====================================================
            # MODEL SELECTION (TRAIN LOSS ONLY)
            # ====================================================
            if train_loss < best_train:
                best_train = train_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        writer.close()

        # ============================================================
        # FINAL TEST USING BEST TRAIN-LOSS MODEL
        # ============================================================
        model.load_state_dict(best_state)
        model.eval()

        probs = []
        with torch.no_grad():
            for xb in X_test:
                probs.append(torch.sigmoid(model(xb.unsqueeze(0).to(device))).item())

        pred = int(np.mean(probs) > 0.5)
        true = video_to_label[test_video]
        proba = float(np.mean(probs))

        all_preds.append(pred)
        all_trues.append(true)
        task_preds.append(pred)
        task_trues.append(true)

        # SAVE VIDEO RESULTS
        fold_results.append({
            "task": task,
            "video_name": test_video.split("/")[-1],
            "video_path": test_video,
            "true_label": true,
            "predicted_label": pred,
            "predicted_probability": proba,
            "correct": int(pred == true)
        })

    if len(task_trues) > 0:
        print(
            f"Task {task} | "
            f"Video-ACC = {accuracy_score(task_trues, task_preds):.3f} | "
            f"Video-F1 = {f1_score(task_trues, task_preds):.3f}"
        )

# ============================================================
# SAVE VIDEO PREDICTIONS TO EXCEL
# ============================================================
predictions_df = pd.DataFrame(fold_results)
predictions_df.to_excel("task_attention_lstm_video_predictions.xlsx", index=False)

print("Saved video predictions to: task_attention_lstm_video_predictions.xlsx")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n=== FINAL RESULTS ===")
print("Accuracy:", accuracy_score(all_trues, all_preds))
print("F1:", f1_score(all_trues, all_preds))
print("Confusion matrix:\n", confusion_matrix(all_trues, all_preds))