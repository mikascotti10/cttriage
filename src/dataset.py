"""
Fase 5 — Dataset y DataLoader
-----------------------------
Carga imágenes (DICOM o PNG), aplica preprocesamiento
y devuelve tensores (X, y) listos para el modelo.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from preprocess import dicom_to_channels, png_to_3ch

# Mapeo de etiquetas
LABEL_MAP = {
    "Normal": 0,
    "Ischemia": 1,
    "Bleeding": 2
}

class CTDataset(Dataset):
    def __init__(self, parquet_path, augment=False):
        """
        Args:
            parquet_path: ruta al archivo train_split.parquet o val_split.parquet
            augment: si aplicar aumentaciones o no
        """
        self.df = pd.read_parquet(parquet_path)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["image_path"])
        label = LABEL_MAP[row["label"]]

        # Cargar imagen según formato
        if row["format"].upper() == "DICOM":
            img = dicom_to_channels(path)
        else:
            img = png_to_3ch(path)

        # Convertir a tensor
        x = torch.tensor(img, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        # (opcional) aumentaciones leves
        if self.augment:
            # ejemplo simple: invertir eje horizontal aleatoriamente
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])

        return x, y


def make_dataloaders(train_path, val_path, batch_size=16):
    """
    Crea DataLoaders para entrenamiento y validación.
    """
    train_ds = CTDataset(train_path, augment=True)
    val_ds = CTDataset(val_path, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dl, val_dl


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    train_pq = BASE_DIR / "train_split.parquet"
    val_pq = BASE_DIR / "val_split.parquet"

    train_dl, val_dl = make_dataloaders(train_pq, val_pq, batch_size=8)
    xb, yb = next(iter(train_dl))
    print("Batch shape:", xb.shape)
    print("Labels:", yb)
