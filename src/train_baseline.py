"""
Fase 6 — Entrenamiento Baseline (entrenamiento completo)
--------------------------------------------------------
Usa ResNet-18 preentrenada en ImageNet,
entrena sobre PNG con ImageFolder y evalúa métricas básicas.
Guarda el mejor checkpoint según macro-F1 y registra
métricas por época en reports/training_log.csv.
Opcional: evalúa External_Test con labels.csv (+ MASKS).
"""

# ============================================================
# 🔹 Importaciones y configuración del entorno
# ============================================================
import sys, csv, time, argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, datasets, transforms

from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Extras para evaluación externa (labels.csv + MASKS)
import pandas as pd
import numpy as np
import cv2

# ============================================================
# 🔹 Utilidades
# ============================================================
def default_transforms(img_size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # PNG 1 canal -> 3 canales
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def make_imagefolder_loaders(data_root: Path, batch_size=16, num_workers=2, pin_memory=True, img_size=224):
    """
    Estructura esperada:
      data_root/
        train/{class}/...png (puede haber subcarpetas como PNG/)
        val/{class}/...png
    """
    tfm = default_transforms(img_size)
    train_ds = datasets.ImageFolder(str(data_root / "train"), transform=tfm)
    val_ds   = datasets.ImageFolder(str(data_root / "val"),   transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    return train_dl, val_dl, train_ds.classes

def build_model(num_classes=3, weights="imagenet"):
    if weights and weights.lower() == "imagenet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# =============== Evaluación External_Test (labels.csv + MASKS) =================

def _read_png(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im

def _mask_bbox(mask, pad=8):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[0]-1, mask.shape[1]-1
    y1, y2 = max(0, ys.min()-pad), min(mask.shape[0]-1, ys.max()+pad)
    x1, x2 = max(0, xs.min()-pad), min(mask.shape[1]-1, xs.max()+pad)
    return y1, x1, y2, x2

class ExternalTestCSV(Dataset):
    """
    Espera:
      base/
        PNG/*.png
        labels.csv con columnas: filename,label
        MASKS/*.png (opcional)
    """
    def __init__(self, base_dir, classes=None, size=224, use_masks=False, roi_mode="mask"):
        self.base = Path(base_dir)
        self.img_dir = self.base / "PNG"
        self.mask_dir = self.base / "MASKS"
        self.use_masks = use_masks and self.mask_dir.exists()
        self.roi_mode = roi_mode
        self.size = size

        df = pd.read_csv(self.base / "labels.csv")
        df = df.dropna(subset=["filename", "label"])
        if classes is None:
            classes = sorted(df["label"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Filtrar a clases conocidas y existentes en disco
        rows = []
        for fn, lb in zip(df["filename"], df["label"]):
            p = self.img_dir / fn
            if lb in self.class_to_idx and p.exists():
                rows.append((p, self.class_to_idx[lb], fn))
        self.items = rows

        # Cache de disponibilidad de máscaras
        self.has_mask = {}
        if self.use_masks:
            for p, _, _ in self.items:
                self.has_mask[p.name] = (self.mask_dir / p.name).exists()

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path, y, fn = self.items[i]
        img = _read_png(img_path).astype(np.float32) / 255.0
        mask = None

        if self.use_masks and self.has_mask.get(img_path.name, False):
            m = _read_png(self.mask_dir / img_path.name)
            m = (m > 0).astype(np.float32)
            mask = m
            if self.roi_mode == "mask":
                img = img * m
            elif self.roi_mode == "crop":
                y1, x1, y2, x2 = _mask_bbox(m, pad=8)
                img = img[y1:y2+1, x1:x2+1]
                m   = m[y1:y2+1, x1:x2+1]
                mask = m

        # resize y formar 3 canales (mantener compatibilidad ImageNet)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = np.stack([img, img, img], axis=0).astype(np.float32)

        return torch.from_numpy(x), y, fn

def evaluate_external(model, device, external_dir, classes, batch_size=32, num_workers=2,
                      use_masks=False, roi_mode="mask", img_size=224):
    """
    Evalúa External_Test usando labels.csv (+ opcional MASKS).
    """
    base = Path(external_dir)
    csv_path = base / "labels.csv"
    if not csv_path.exists():
        print(f"  No hay labels.csv en {external_dir}; se omite evaluación externa.")
        return None

    ds = ExternalTestCSV(base, classes=classes, size=img_size,
                         use_masks=use_masks, roi_mode=roi_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    all_y, all_p = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, _fns in tqdm(dl, desc="External_Test"):
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            all_p.extend(preds.tolist())
            all_y.extend(yb.numpy().tolist())

    macro = f1_score(all_y, all_p, average="macro")
    print(f"\n External_Test — Macro-F1: {macro:.4f}")
    print(classification_report(all_y, all_p, target_names=classes))
    print("Matriz de confusión:\n", confusion_matrix(all_y, all_p))
    return macro

# ============================================================
# 🔹 Entrenamiento
# ============================================================
def train_baseline(
    data_root: str = None,
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 1e-4,
    num_workers: int = 2,
    pin_memory: bool = True,
    weights: str = "imagenet",
    img_size: int = 224,
    eval_external: bool = False,
    external_dir: str = "/content/data_local/External_Test",
    use_masks: bool = False,
    roi_mode: str = "mask",  # 'mask'|'crop'|'none'
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_DIR = Path(__file__).resolve().parent.parent

    # DATA ROOT: env > arg > default
    data_root = Path(
        data_root
        or sys.environ.get("DATA_ROOT", "/content/data_split")
    )

    # -------------------------------
    # DATA
    # -------------------------------
    train_dl, val_dl, classes = make_imagefolder_loaders(
        data_root, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, img_size=img_size
    )
    print(f"Clases: {classes}")
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")
    print("DEVICE:", DEVICE, "| CUDA available:", torch.cuda.is_available())

    # -------------------------------
    # MODELO
    # -------------------------------
    model = build_model(num_classes=len(classes), weights=weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
    # ENTRENAMIENTO
    # -------------------------------
    for epoch in range(epochs):
        epoch_t0 = time.time()
        model.train()
        total_loss = 0.0

        for xb, yb in tqdm(train_dl, desc=f"Train epoch {epoch+1}/{epochs}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_loss:.4f}")

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
        print(f"Validation macro-F1: {val_f1:.4f}")

        # Log CSV
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                round(avg_loss, 4),
                round(val_f1, 4),
                round(time.time() - epoch_t0, 2),
            ])

        # Checkpoint por F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f" Nuevo mejor modelo (F1={best_f1:.4f}) guardado en {best_path}")

    print(" Entrenamiento finalizado.")
    print(f" Log guardado en: {log_file}")
    print(f" Mejor modelo: {best_path} (F1={best_f1:.4f})")

    # -------------------------------
    # EVALUACIÓN EXTERNAL_TEST (opcional)
    # -------------------------------
    if eval_external:
        print("\n=== Evaluación en External_Test ===")
        _ = evaluate_external(
            model, DEVICE,
            external_dir=external_dir,
            classes=classes,
            batch_size=max(32, batch_size),
            num_workers=num_workers,
            use_masks=use_masks,
            roi_mode=roi_mode,
            img_size=img_size,
        )

# ============================================================
# 🔹 CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None, help="Raíz con train/ y val/ (por defecto $DATA_ROOT o /content/data_split)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", type=int, default=1)
    p.add_argument("--weights", type=str, default="imagenet", choices=["imagenet","none"])
    p.add_argument("--img-size", type=int, default=224)

    p.add_argument("--eval-external", action="store_true")
    p.add_argument("--external-dir", type=str, default="/content/data_local/External_Test")
    p.add_argument("--use-masks", action="store_true")
    p.add_argument("--roi-mode", type=str, default="mask", choices=["mask","crop","none"])
    return p.parse_args()

# ============================================================
# 🔹 Main
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    train_baseline(
        data_root=args.data_root,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        weights=args.weights,
        img_size=args.img_size,
        eval_external=args.eval_external,
        external_dir=args.external_dir,
        use_masks=args.use_masks,
        roi_mode=args.roi_mode,
    )
