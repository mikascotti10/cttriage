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

# Identificador de grupo (por estudio o por carpeta padre)
# Si tenés una columna con el ID de estudio, usala aquí:
if "study" in df.columns:
    groups = df["study"]
else:
    # si no existe, inferimos el grupo a partir del directorio del archivo
    groups = df["image_path"].apply(lambda x: Path(x).parent.name)

# Definir el divisor: 80% train, 20% val
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(splitter.split(df, groups=groups))

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

print(f"Train: {len(train_df)} imágenes ({len(train_df)/len(df):.1%})")
print(f"Val:   {len(val_df)} imágenes ({len(val_df)/len(df):.1%})")

# Guardar splits
train_df.to_parquet(BASE_DIR / "train_split.parquet", index=False)
val_df.to_parquet(BASE_DIR / "val_split.parquet", index=False)
print("✅ Guardados train_split.parquet y val_split.parquet")
