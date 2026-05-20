# Herramienta de apoyo a la toma de decisiones para la descarbonización industrial

Este repositorio mantiene intacta la app original en Streamlit y añade una nueva implementación paralela con:

- `web-studio-v2/backend/` en `FastAPI`
- `web-studio-v2/frontend/` en `React + Vite + TypeScript`

La app de Streamlit sigue viviendo en `app.py` y no se usa para esta nueva interfaz.

## Estructura

```text
web-studio-v2/
  backend/
    main.py
    app_logic.py
    scope2_electricity.py
    data/
  frontend/
    src/
docker-compose.yml
```

## Desarrollo local

### Opción 1. Docker Compose

1. Opcional: crea `web-studio-v2/backend/.env` a partir de `web-studio-v2/backend/.env.example` si vas a usar claves locales.
2. Opcional: crea `web-studio-v2/frontend/.env` a partir de `web-studio-v2/frontend/.env.example` si necesitas sobrescribir la URL del backend.
3. Ejecuta:

```bash
docker-compose up --build
```

Servicios:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

### Opción 2. Manual

Backend:

```bash
cd web-studio-v2/backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd web-studio-v2/frontend
npm install
npm run dev
```

## Variables de entorno

Backend:

- `GEMINI_API_KEY` opcional
- `GEMINI_MODEL` opcional, por defecto `gemini-2.5-flash`
- `BACKEND_CORS_ORIGINS` opcional

Frontend:

- `VITE_API_BASE_URL` apuntando al backend desplegado

## Endpoints

- `POST /api/calculate-footprint`
- `POST /api/generate-pestel`
- `POST /api/generate-initiatives`
- `POST /api/generate-ai-initiatives`
- `POST /api/compute-metrics`
- `POST /api/optimize-portfolio`
- `GET /api/catalogs/stationary-fuels`
- `GET /api/catalogs/mobile-fuels`
- `GET /api/catalogs/refrigerants`
- `GET /api/catalogs/electricity-suppliers?year=2024`

## Despliegue gratuito

### Backend en Render o Railway

1. Crea un servicio desde la carpeta `web-studio-v2/backend/`
2. Usa el `Dockerfile` incluido o el comando:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Configura `GEMINI_API_KEY` si quieres IA real

### Frontend en Vercel o Netlify

1. Publica la carpeta `web-studio-v2/frontend/`
2. Define `VITE_API_BASE_URL` con la URL pública del backend
3. En Vercel ya está incluido `vercel.json` para SPA fallback

## Notas

- `web-studio-v2/backend/scope2_electricity.py` está copiado como referencia sin modificar.
- La lógica utilizable por API vive en `web-studio-v2/backend/app_logic.py`, sin dependencias de `streamlit`.
- Los catálogos CSV se cargan una vez y quedan cacheados en memoria.
