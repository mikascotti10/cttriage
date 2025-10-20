from pathlib import Path
import pydicom, pandas as pd, PIL.Image as Image

ROOT = Path(".")  
DATA = ROOT/"data"  

def dicom_tags(fp):
    try:
        ds = pydicom.dcmread(fp, stop_before_pixels=True)
        return {
            "study_id": getattr(ds, "StudyInstanceUID", None),
            "series_id": getattr(ds, "SeriesInstanceUID", None),
            "instance": getattr(ds, "InstanceNumber", None),
            "modality": getattr(ds, "Modality", None),
            "wl": getattr(ds, "WindowCenter", None),
            "ww": getattr(ds, "WindowWidth", None),
            "slope": float(getattr(ds, "RescaleSlope", 1.0)),
            "intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
        }
    except Exception:
        return {"study_id": None,"series_id": None,"instance": None,"modality": None,
                "wl": None,"ww": None,"slope": None,"intercept": None}

def png_size(fp):
    try:
        w,h = Image.open(fp).size
        return w,h
    except Exception:
        return None,None

def scan():
    rows = []
    # carpetas esperadas: Bleeding/, Ischemia/, Normal/, External_Test/
    for cls in ["Bleeding","Ischemia","Normal","External_Test"]:
        d = DATA/cls
        if not d.exists(): continue
        for fp in d.rglob("*"):
            if fp.is_dir(): continue
            fmt = "DICOM" if fp.suffix.lower()==".dcm" else ("PNG" if fp.suffix.lower()==".png" else None)
            if fmt is None: continue
            label = cls if cls!="External_Test" else None
            width=height=channels=None
            study_id=series_id=instance=None
            slope=intercept=wl=ww=None
            if fmt=="DICOM":
                tags = dicom_tags(fp)
                study_id,series_id,instance = tags["study_id"],tags["series_id"],tags["instance"]
                slope,intercept,wl,ww = tags["slope"],tags["intercept"],tags["wl"],tags["ww"]
            else:
                width,height = png_size(fp)
            rows.append({
                "image_path": str(fp), "label": label, "subset": "external" if cls=="External_Test" else "train",
                "format": fmt, "width": width, "height": height, "study_id": study_id, "series_id": series_id,
                "instance": instance, "slope": slope, "intercept": intercept, "wl": wl, "ww": ww
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = scan()
    out = ROOT / "samples.parquet"

    # --- Normalizar columnas wl y ww ---
    def normalize_multivalue(x):
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            try:
                return float(list(x)[0])
            except Exception:
                return None
        return x

    for col in ["wl", "ww"]:
        df[col] = df[col].apply(normalize_multivalue)

    df.to_parquet(out, index=False)
    print(f"Escribí {out} con {len(df)} filas")
