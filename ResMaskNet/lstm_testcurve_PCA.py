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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# MODEL
# ============================================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.4):
        super().__init__()

        self.norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True
        )

        self.hnorm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.norm(x)

        _, (hn, _) = self.lstm(x)

        h = hn[-1]

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

video_to_chunks = {}
video_to_label = {}
video_to_task = {}

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

            chunks.append(
                torch.tensor(
                    emb[start:end],
                    dtype=torch.float32
                )
            )

            labels.append(label)

            video_to_chunks.setdefault(video, []).append(i)

            i += 1

            break

        chunks.append(
            torch.tensor(
                emb[start:end],
                dtype=torch.float32
            )
        )

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
def train_one(
    train_loader,
    input_size,
    run_name,
    scale,
    test_videos,
    video_to_chunks,
    video_to_label,
    max_epochs=100,
    patience=40
):

    model = SimpleLSTM(input_size=input_size).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=1,
        eta_min=1e-5
    )

    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir=run_name)

    best_train_loss = float("inf")
    best_state = None

    epochs_no_improve = 0

    for epoch in range(max_epochs):

        # ---------------- TRAIN ----------------
        model.train()

        train_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            loss = criterion(model(xb), yb)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * len(yb)

        train_loss /= len(train_loader.dataset)

        scheduler.step(epoch)

        # ---------------- TEST ACC ----------------
        model.eval()

        test_preds = []
        test_trues = []

        with torch.no_grad():

            for v in test_videos:

                chunk_preds = []

                for idx in video_to_chunks[v]:

                    xb = scale(idx).unsqueeze(0).to(device)

                    p = torch.sigmoid(model(xb)).item()

                    chunk_preds.append(p > 0.5)

                pred_video = int(
                    sum(chunk_preds) > len(chunk_preds) / 2
                )

                test_preds.append(pred_video)

                test_trues.append(video_to_label[v])

        test_acc = accuracy_score(test_trues, test_preds)

        # ---------------- LOGGING ----------------
        writer.add_scalar("Loss/train", train_loss, epoch)

        writer.add_scalar("Accuracy/test", test_acc, epoch)

        writer.add_scalar(
            "LR",
            optimizer.param_groups[0]["lr"],
            epoch
        )

        print(
            f"{run_name} | Epoch {epoch:03d} | "
            f"TrainLoss={train_loss:.4f} | "
            f"TestAcc={test_acc:.4f}"
        )

        # ---------------- EARLY STOP ----------------
        if train_loss < best_train_loss:

            best_train_loss = train_loss

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            epochs_no_improve = 0

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    writer.close()

    model.load_state_dict(best_state)

    model.to(device)

    return model

# ============================================================
# TEST MULTIPLE PCA DIMENSIONS
# ============================================================

pca_dims = [3, 8, 16, 32]

results = {}

# NEW: SAVE VIDEO PREDICTIONS
prediction_rows = []

for n_comp in pca_dims:

    print(f"\n\n==============================")
    print(f"PCA DIMENSIONS: {n_comp}")
    print(f"==============================")

    all_video_preds = []
    all_video_trues = []

    for test_subject in subjects:

        train_subjects = [
            s for s in subjects
            if s != test_subject
        ]

        idx_train = []

        for s in train_subjects:

            for v in subject_to_videos[s]:

                idx_train.extend(video_to_chunks[v])

        test_videos = subject_to_videos[test_subject]

        # ---------------- STANDARDIZATION ----------------
        scaler = StandardScaler().fit(
            X[idx_train].reshape(-1, X.shape[2])
        )

        X_scaled = scaler.transform(
            X.reshape(-1, X.shape[2])
        )

        X_scaled = X_scaled.reshape(X.shape)

        # ---------------- PCA ----------------
        pca = PCA(n_components=n_comp)

        X_train_2d = X_scaled[idx_train].reshape(
            -1,
            X.shape[2]
        )

        pca.fit(X_train_2d)

        X_pca_2d = pca.transform(
            X_scaled.reshape(-1, X.shape[2])
        )

        X_pca = X_pca_2d.reshape(
            X.shape[0],
            X.shape[1],
            n_comp
        )

        # ---------------- SCALE FUNCTION ----------------
        def scale(idx):

            return torch.tensor(
                X_pca[idx],
                dtype=torch.float32
            )

        train_loader = DataLoader(
            TensorDataset(
                scale(idx_train),
                y[idx_train]
            ),
            batch_size=32,
            shuffle=True
        )

        run_name = f"runs/pca_{n_comp}/subject_{test_subject}"

        model = train_one(
            train_loader,
            input_size=n_comp,
            run_name=run_name,
            scale=scale,
            test_videos=test_videos,
            video_to_chunks=video_to_chunks,
            video_to_label=video_to_label
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

            # SAME MAJORITY VOTING AS BEFORE
            pred_video = int(
                sum(chunk_preds) > len(chunk_preds) / 2
            )

            true_video = video_to_label[v]

            mean_probability = float(np.mean(chunk_probs))

            all_video_preds.append(pred_video)

            all_video_trues.append(true_video)

            # NEW: SAVE ROW
            prediction_rows.append({
                "pca_dimensions": n_comp,
                "subject": test_subject,
                "task": video_to_task[v],
                "video": v,
                "true_label": true_video,
                "predicted_label": pred_video,
                "mean_probability": mean_probability,
                "correct": int(pred_video == true_video)
            })

    acc = accuracy_score(
        all_video_trues,
        all_video_preds
    )

    f1 = f1_score(
        all_video_trues,
        all_video_preds
    )

    results[n_comp] = (acc, f1)

    print(f"\nRESULTS: Acc={acc:.3f} | F1={f1:.3f}")

# ============================================================
# SAVE EXCEL
# ============================================================

predictions_df = pd.DataFrame(prediction_rows)

predictions_df.to_excel(
    "lstm_pca_video_predictions.xlsx",
    index=False
)

print("\nSaved predictions to:")
print(" - lstm_pca_video_predictions.xlsx")

# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n===== FINAL COMPARISON =====")

for n, (acc, f1) in results.items():

    print(
        f"PCA {n:>2} dims -> "
        f"Acc: {acc:.3f} | "
        f"F1: {f1:.3f}"
    )