# GaziGoz Team EDI OCT Streamlit App

This repository contains the Streamlit app for EDI OCT segmentation and measurement.

## Local Run

Use a Python environment with the packages from `requirements.txt`, then run:

```bash
streamlit run app.py
```

## Deploy To Streamlit Community Cloud

### Important

The trained `.keras` model files are larger than GitHub's normal file size limit, so they should **not** be committed directly into the repository.

Instead, keep the app code in GitHub and provide the model through a downloadable URL.

## Recommended GitHub Setup

1. Push this app code to your GitHub repository.
2. Create a GitHub Release in that repository.
3. Upload your chosen model file, for example:
   - `complex_unet_edi_best.keras`
4. Copy the direct download URL for that release asset.

Example release asset URL format:

```text
https://github.com/emirdoogan/gazigozteam/releases/download/v1/complex_unet_edi_best.keras
```

## Streamlit Cloud Secrets

In Streamlit Community Cloud, add these secrets:

```toml
MODEL_URL = "https://github.com/emirdoogan/gazigozteam/releases/download/v1/complex_unet_edi_best.keras"
MODEL_FILENAME = "complex_unet_edi_best.keras"
```

The app will:

- first look for a local `.keras` model file
- if no local file exists, download the model from `MODEL_URL`

## Files To Keep In The Repo

- `app.py`
- `raw_preprocessing.py`
- `simple_unet.py`
- `requirements.txt`
- `README.md`

## Files Not Recommended For The Repo

- training images
- masks
- logs
- large model artifacts committed directly to git
