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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict

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
        return torch.sum(weights * lstm_out, dim=1)

# ============================================================
# MODEL
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

# ============================================================
# GROUP BY TASK
# ============================================================
task_to_videos = defaultdict(list)
for v in video_to_chunks:
    task_to_videos[video_to_task[v]].append(v)

# ============================================================
# PCA EXPERIMENT
# ============================================================
pca_dims = [8, 16]

# NEW: STORE ALL VIDEO PREDICTIONS
all_prediction_rows = []

for n_comp in pca_dims:

    print(f"\n===== PCA DIM: {n_comp} =====")

    all_preds, all_trues = [], []

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

            # ---------------- STANDARDIZATION ----------------
            scaler = StandardScaler().fit(X_train.reshape(-1, X.shape[2]))

            X_scaled = scaler.transform(X.reshape(-1, X.shape[2]))
            X_scaled = X_scaled.reshape(X.shape)

            # ---------------- PCA ----------------
            pca = PCA(n_components=n_comp)
            X_train_2d = X_scaled[train_idx].reshape(-1, X.shape[2])
            pca.fit(X_train_2d)

            X_pca_2d = pca.transform(X_scaled.reshape(-1, X.shape[2]))
            X_pca = X_pca_2d.reshape(X.shape[0], X.shape[1], n_comp)

            X_train = torch.tensor(X_pca[train_idx], dtype=torch.float32)
            X_test  = torch.tensor(X_pca[test_idx], dtype=torch.float32)

            # ---------------- LOADER ----------------
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=8,
                shuffle=True
            )

            # ---------------- MODEL ----------------
            model = AttentionLSTM(input_size=n_comp).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=1, eta_min=1e-5
            )

            criterion = nn.BCEWithLogitsLoss()

            writer = SummaryWriter(
                log_dir=f"runs/task_attention_lstm_pca/{n_comp}/{task}/{test_video.replace('/', '_')}"
            )

            best_train = float("inf")
            best_state = None
            counter = 0
            patience = 40

            # ---------------- TRAIN ----------------
            for epoch in range(100):

                model.train()
                total_loss = 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * yb.size(0)

                train_loss = total_loss / len(train_loader.dataset)
                scheduler.step(epoch)

                # TEST (curve only)
                model.eval()
                probs = []
                with torch.no_grad():
                    for xb in X_test:
                        probs.append(torch.sigmoid(model(xb.unsqueeze(0).to(device))).item())

                pred_epoch = int(np.mean(probs) > 0.5)
                true_epoch = video_to_label[test_video]
                test_acc = int(pred_epoch == true_epoch)

                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Accuracy/test", test_acc, epoch)

                print(f"Epoch {epoch+1:03d} | TrainLoss={train_loss:.4f} | TestAcc={test_acc}")

                if train_loss < best_train:
                    best_train = train_loss
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break

            writer.close()

            # ---------------- FINAL TEST ----------------
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

            # SAVE VIDEO RESULT
            all_prediction_rows.append({
                "pca_dimensions": n_comp,
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

    # ---------------- FINAL ----------------
    print("\n=== FINAL RESULTS ===")
    print("Accuracy:", accuracy_score(all_trues, all_preds))
    print("F1:", f1_score(all_trues, all_preds))
    print("Confusion matrix:\n", confusion_matrix(all_trues, all_preds))

# ============================================================
# SAVE VIDEO PREDICTIONS TO EXCEL
# ============================================================
predictions_df = pd.DataFrame(all_prediction_rows)
predictions_df.to_excel("attention_lstm_pca_task_video_predictions.xlsx", index=False)

print("Saved predictions to: attention_lstm_pca_task_video_predictions.xlsx")