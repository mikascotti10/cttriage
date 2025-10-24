"""
Fase 4 — Split del dataset sin fuga entre estudios
-------------------------------------------------
Crea train/val asegurando que todas las imágenes
del mismo estudio queden en el mismo conjunto.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# Ruta base
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "samples.parquet"

# Leer la tabla del dataset
df = pd.read_parquet(DATASET)
print("Total de imágenes:", len(df))

# 🔹 Normalizar etiquetas
if "label" in df.columns:
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()

# 🔹 Filtrar etiquetas válidas
valid_labels = ["Normal", "Ischemia", "Bleeding"]
df = df[df["label"].isin(valid_labels)]
df = df[df["label"].notnull()]

# 🔹 Filtrar subset (solo train)
if "subset" in df.columns:
    df["subset"] = df["subset"].astype(str).str.lower().str.strip()
    df = df[df["subset"] == "train"]

print("Imágenes con etiqueta válida (subset=train):", len(df))
print("Clases encontradas:\n", df["label"].value_counts())

# 🔹 Determinar grupos (evita NoneType)
if "study_id" in df.columns and df["study_id"].notnull().any():
    groups = df["study_id"]
else:
    # fallback: usar carpeta padre del path
    groups = df["image_path"].apply(lambda x: Path(x).parent.name)

# 🔹 Split 80/20 por grupos
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(splitter.split(df, groups=groups))

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

print(f"Train: {len(train_df)} imágenes ({len(train_df)/len(df):.1%})")
print(f"Val:   {len(val_df)} imágenes ({len(val_df)/len(df):.1%})")

# 🔹 Guardar splits
train_path = BASE_DIR / "train_split.parquet"
val_path = BASE_DIR / "val_split.parquet"
train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
print("✅ Guardados train_split.parquet y val_split.parquet")
