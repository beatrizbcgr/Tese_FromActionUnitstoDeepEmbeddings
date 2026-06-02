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
# LSTM MODEL
# ============================================================
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=2, dropout=0.3):
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
        h = hn[-1]
        h = self.hidden_norm(h)
        return self.fc(h).squeeze(-1)

# ============================================================
# LOAD DATA + CHUNKS
# ============================================================
with open("embeddings_and_predictions.pkl", "rb") as f:
    data = pickle.load(f)

all_lengths = [len(info["embeddings"]) for info in data.values()]
min_len = min(all_lengths)
print("Min sequence length:", min_len)

def extract_task(video):
    parts = video.split("/")[-1].split("_")
    return parts[2] + "_" + parts[3]

chunks, labels = [], []
video_to_chunks, video_to_label, video_to_task = {}, {}, {}

i = 0
SKIP_TASKS = {"NSM_BIGSMILE", "NSM_BROW"}
OVERLAP = 0.30
stride = int(min_len * (1 - OVERLAP))

for video, info in data.items():
    emb = np.array(info["embeddings"])
    label = 1 if "stroke" in video.lower() else 0
    task = extract_task(video)

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

X = torch.stack(chunks)
y = torch.tensor(labels, dtype=torch.float32)

# ============================================================
# GROUP BY TASK
# ============================================================
task_to_videos = defaultdict(list)
for v in video_to_chunks:
    task_to_videos[video_to_task[v]].append(v)

# ============================================================
# PCA SETTINGS
# ============================================================
pca_dims = [8, 16]

# ============================================================
# STORE ALL PREDICTIONS
# ============================================================
all_prediction_rows = []

# ============================================================
# MAIN LOOP
# ============================================================
for n_comp in pca_dims:

    print(f"\n\n===== PCA DIM: {n_comp} =====")

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

            # STANDARDIZATION
            scaler = StandardScaler().fit(
                X_train.reshape(-1, X.shape[2])
            )

            X_train = scaler.transform(X_train.reshape(-1, X.shape[2]))
            X_test  = scaler.transform(X_test.reshape(-1, X.shape[2]))

            # PCA
            pca = PCA(n_components=n_comp)
            pca.fit(X_train)

            X_train = pca.transform(X_train)
            X_test  = pca.transform(X_test)

            X_train = torch.tensor(
                X_train.reshape(len(train_idx), min_len, n_comp),
                dtype=torch.float32
            )

            X_test = torch.tensor(
                X_test.reshape(len(test_idx), min_len, n_comp),
                dtype=torch.float32
            )

            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=8,
                shuffle=True
            )

            model = ImprovedLSTM(input_size=n_comp).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=1, eta_min=1e-5
            )

            criterion = nn.BCEWithLogitsLoss()

            best_train = float("inf")
            best_state = None
            counter = 0
            patience = 40

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

                if train_loss < best_train:
                    best_train = train_loss
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break

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

            # ====================================================
            # SAVE PREDICTIONS
            # ====================================================
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

        print(
            f"Task {task} | "
            f"Acc={accuracy_score(task_trues, task_preds):.3f} | "
            f"F1={f1_score(task_trues, task_preds):.3f}"
        )

    # overall
    print("\n=== FINAL RESULTS ===")
    print(f"PCA {n_comp} | Accuracy:", accuracy_score(all_trues, all_preds))
    print(f"PCA {n_comp} | F1:", f1_score(all_trues, all_preds))
    print("Confusion matrix:\n", confusion_matrix(all_trues, all_preds))

# ============================================================
# SAVE PREDICTIONS TO EXCEL
# ============================================================
predictions_df = pd.DataFrame(all_prediction_rows)
predictions_df.to_excel("lstm_pca_task_video_predictions.xlsx", index=False)

print("\nSaved predictions to: lstm_pca_task_video_predictions.xlsx")