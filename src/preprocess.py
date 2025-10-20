import numpy as np
import pydicom
import cv2
from pathlib import Path

# ---------------------------
# 1. Conversión a Hounsfield
# ---------------------------
def to_hu(ds):
    """
    Convierte un objeto pydicom a Unidades Hounsfield.
    HU = pixel * RescaleSlope + RescaleIntercept
    """
    arr = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


# ---------------------------
# 2. Aplicación de ventana
# ---------------------------
def window(img_hu, wl, ww):
    """
    Aplica una ventana (window level, window width).
    Devuelve imagen normalizada en [0,1].
    """
    low, high = wl - ww / 2.0, wl + ww / 2.0
    clipped = np.clip(img_hu, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


# ---------------------------
# 3. Pipeline DICOM → multi-canal
# ---------------------------
def dicom_to_channels(path, windows=((40, 80), (50, 130)), out_size=224):
    import numpy as np, cv2, pydicom
    ds = pydicom.dcmread(path)
    hu = to_hu(ds)
    chans = [window(hu, wl, ww) for wl, ww in windows]
    chans = [cv2.resize(c, (out_size, out_size), interpolation=cv2.INTER_AREA)
             for c in chans]

    x = np.stack(chans, axis=0)  # (2,H,W)
    # Si hay solo 2 canales, agregamos un tercero (por ejemplo repitiendo el primero)
    if x.shape[0] == 2:
        x = np.concatenate([x, x[:1]], axis=0)
    return x



# ---------------------------
# 4. Pipeline PNG → 3 canales
# ---------------------------
def png_to_3ch(path, out_size=224):
    """
    Convierte PNG a 3 canales replicados, normalizado [0,1].
    """
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (out_size, out_size), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    return np.stack([im, im, im], axis=0)  # (3,H,W)


# ---------------------------
# 5. Prueba manual
# ---------------------------
if __name__ == "__main__":
    # --- Detectar automáticamente la carpeta base del proyecto ---
    BASE_DIR = Path(__file__).resolve().parent.parent   # sube desde /src hasta /cttriage
    DATA_DIR = BASE_DIR / "data"

    sample_dcm = DATA_DIR / "Bleeding"
    sample_png = DATA_DIR / "Normal"

    # Buscar archivos
    dcm_files = list(sample_dcm.rglob("*.dcm"))
    png_files = list(sample_png.rglob("*.png"))

    print("📂 Carpeta DICOM:", sample_dcm)
    print("📂 Carpeta PNG:", sample_png)
    print("🧮 DICOM encontrados:", len(dcm_files))
    print("🧮 PNG encontrados:", len(png_files))

    # Pruebas
    if dcm_files:
        x = dicom_to_channels(dcm_files[0])
        print("✅ DICOM →", x.shape, x.min(), x.max())
    else:
        print("⚠️ No se encontraron archivos DICOM en", sample_dcm)

    if png_files:
        y = png_to_3ch(png_files[0])
        print("✅ PNG →", y.shape, y.min(), y.max())
    else:
        print("⚠️ No se encontraron archivos PNG en", sample_png)
