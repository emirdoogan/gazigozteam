# GaziGoz Team EDI OCT Streamlit App

Streamlit app for EDI OCT segmentation and measurement (496x496 U-Net, 13 retinal
layers) running on ONNX Runtime.

## Repository Contents

- `app.py` - Streamlit application
- `raw_preprocessing.py` - cropping and preprocessing helpers
- `simple_unet.py` - `GroupNormalization` layer for the TensorFlow/Keras path
- `requirements.txt` - ONNX Runtime dependency set used by the deployment
- `complex_unet_edi_best.onnx` - segmentation model (38 MB)
- `README.md`

Training notebooks and annotation JSONs live outside this repository.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app prefers the ONNX model whenever `onnxruntime` is installed, so no
TensorFlow is needed for the deployed path.

## Deploy To Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io and sign in with GitHub.
3. Create a new app with:
   - Repository: this repository
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: `3.11`
4. Deploy.

No secrets or configuration are required because the ONNX model is included in
the repository. The app is then reachable at `https://<app-name>.streamlit.app`.

## Model Access

- a local `.onnx` model in the project root is used first
- otherwise a local `.keras` model is used (requires TensorFlow)
- otherwise the app falls back to `MODEL_URL` / `MODEL_FILENAME` from secrets
