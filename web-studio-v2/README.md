# Web Studio V2

Implementación web paralela para comparar una versión más profesional de la herramienta frente a la app original en Streamlit.

## Principios

- La app original en Streamlit no se modifica.
- El frontend y el backend existentes fuera de Streamlit tampoco se tocan.
- `web-studio-v2` reutiliza la lógica, los catálogos y la secuencia funcional de la herramienta actual, pero en una experiencia visual distinta.

## Estructura

```text
web-studio-v2/
  backend/
  frontend/
```

## Backend

```bash
cd web-studio-v2/backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8010
```

## Frontend

```bash
cd web-studio-v2/frontend
npm install
npm run dev
```

## URLs

- Streamlit original: la que ya uses para `app.py`
- Web Studio V2 backend: `http://127.0.0.1:8010`
- Web Studio V2 frontend: `http://127.0.0.1:5180`

## Cobertura funcional

- Landing explicativa inicial.
- Inputs de empresa, Alcance 1, Alcance 2 y supuestos financieros.
- Cálculo de huella de carbono.
- Análisis PESTEL con IA.
- Generación y edición de iniciativas.
- Cálculo de métricas financieras.
- Optimización de portfolio.
- Exportación CSV del portfolio resultante.

## Nota

Este proyecto está pensado para evolucionar visualmente sin comprometer la referencia funcional actual en Streamlit.
