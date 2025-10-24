"""
Fase 6 — Entrenamiento Baseline (entrenamiento completo)
--------------------------------------------------------
Usa ResNet-18 preentrenada en ImageNet,
entrena sobre CTDataset y evalúa métricas básicas.
Guarda el mejor checkpoint según macro-F1 y registra
métricas por época en reports/training_log.csv.
"""

# ============================================================
# 🔹 Importaciones y configuración del entorno
# ============================================================
import sys
from pathlib import Path
from tqdm import tqdm
import csv, time

# Asegurar import desde la raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import f1_score
from src.dataset import make_dataloaders


# ============================================================
# 🔹 Función principal de entrenamiento
# ============================================================
def train_baseline():
    # -------------------------------
    # CONFIGURACIÓN
    # -------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 1e-4

    BASE_DIR = Path(__file__).resolve().parent.parent
    train_pq = BASE_DIR / "train_split.parquet"
    val_pq = BASE_DIR / "val_split.parquet"

    # -------------------------------
    # DATA
    # -------------------------------
    train_dl, val_dl = make_dataloaders(
        train_pq, val_pq,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False
    )
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")
    print("DEVICE:", DEVICE, "| CUDA available:", torch.cuda.is_available())

    # -------------------------------
    # MODELO
    # -------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -------------------------------
    # LOGGING
    # -------------------------------
    report_dir = BASE_DIR / "reports"
    report_dir.mkdir(exist_ok=True)
    log_file = report_dir / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_macro_f1", "time_sec"])

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    best_f1 = 0.0
    best_path = model_dir / "baseline_resnet18_best.pth"

    # -------------------------------
    # ENTRENAMIENTO COMPLETO
    # -------------------------------
    for epoch in range(EPOCHS):
        epoch_t0 = time.time()
        model.train()
        total_loss = 0

        for xb, yb in tqdm(train_dl, desc=f"Train epoch {epoch+1}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_loss:.4f}")

        # -------------------------------
        # VALIDACIÓN
        # -------------------------------
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in tqdm(val_dl, desc="Validating"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Validation macro-F1: {val_f1:.3f}")

        # Guardar métricas en CSV
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                round(avg_loss, 4),
                round(val_f1, 4),
                round(time.time() - epoch_t0, 2)
            ])

        # Guardar mejor modelo según F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"🌟 Nuevo mejor modelo guardado (F1={best_f1:.3f})")

    # -------------------------------
    # FINAL
    # -------------------------------
    print("✅ Entrenamiento completo finalizado.")
    print(f"📈 Log guardado en: {log_file}")
    print(f"🏆 Mejor modelo: {best_path} (F1={best_f1:.3f})")


# ============================================================
# 🔹 Ejecución segura (requerido en Windows)
# ============================================================
if __name__ == "__main__":
    train_baseline()
