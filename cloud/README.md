# Cloud deployment

Esta carpeta está pensada para desplegar en la nube sin afectar la versión local.

## Streamlit Cloud
- Archivo principal: `cloud/app.py`
- Dependencias: `cloud/requirements.txt`
- Secrets (Streamlit Cloud → App settings → Secrets):
  ```toml
  GEMINI_API_KEY = "TU_NUEVA_API_KEY"
  ```
