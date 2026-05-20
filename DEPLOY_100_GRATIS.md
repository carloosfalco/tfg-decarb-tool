# Propuesta de despliegue 100% gratis

Este documento no modifica la aplicación actual. Es una guía paralela para una alternativa de despliegue con más libertad de frontend sin afectar a `app.py`, `cloud/app.py` ni al despliegue existente.

## Objetivo

Mantener la lógica de cálculo actual en Python y, en una fase posterior, separar:

- `frontend` en `Next.js`
- `backend` en `FastAPI`

Todo ello usando servicios con plan gratuito para un prototipo o TFG.

## Opción recomendada 100% gratis

La combinación más razonable para no pagar al principio es:

- `Next.js` desplegado en `Vercel Hobby`
- `FastAPI` desplegado en `Render Free`
- Repositorio en `GitHub`

## Qué ganas con esta opción

- Mucha más libertad visual que con `Streamlit`
- Mejor control de navegación, layout y responsive
- Posibilidad de diseñar una landing y una experiencia más parecida a un producto real
- Separación limpia entre interfaz y lógica de negocio

## Limitaciones de la opción gratis

- `Render Free` puede dormir el backend tras un tiempo sin uso
- El primer acceso tras inactividad puede tardar unos segundos
- `Vercel Hobby` es suficiente para TFG y demo, pero no está pensado para un SaaS comercial serio
- Si el uso crece, tocará pasar a planes de pago o cambiar infraestructura

## Arquitectura propuesta

### 1. Frontend

`Next.js` se encargaría de:

- Landing de presentación
- Formularios multi-step
- Tablas y visualizaciones
- Gestión de estado de la interfaz
- Llamadas al backend mediante API HTTP

Despliegue recomendado:

- `Vercel Hobby`

### 2. Backend

`FastAPI` se encargaría de:

- Cálculo de huella
- Optimización con `PuLP`
- Generación de iniciativas
- Integración con IA
- Validaciones y transformación de datos

Despliegue recomendado:

- `Render Free`

### 3. Código actual reutilizable

De la app actual conviene reutilizar:

- lógica de cálculo
- factores y datasets en `data/`
- funciones de negocio
- integración con modelos

Lo que no conviene arrastrar tal cual es la capa `Streamlit`, porque es precisamente lo que limita el diseño.

## Ruta de migración sin romper nada

### Fase 1

Mantener la app actual como está y crear una rama o carpeta nueva para la nueva arquitectura.

### Fase 2

Extraer del archivo actual la lógica de negocio a módulos Python reutilizables, por ejemplo:

- `backend/services/emissions.py`
- `backend/services/optimization.py`
- `backend/services/ai.py`

### Fase 3

Crear una API con `FastAPI`, por ejemplo:

- `POST /calculate-footprint`
- `POST /generate-initiatives`
- `POST /optimize-portfolio`
- `POST /generate-brief`

### Fase 4

Crear el frontend con `Next.js` consumiendo esa API.

### Fase 5

Desplegar:

- frontend en `Vercel`
- backend en `Render`

## Coste estimado

Para un TFG o prototipo:

- `GitHub`: gratis
- `Next.js`: gratis
- `FastAPI`: gratis
- `Vercel Hobby`: gratis
- `Render Free`: gratis

Coste total inicial:

- `0 EUR/mes`

## Recomendación práctica

Si el objetivo es terminar el TFG con buena presentación visual pero sin riesgo innecesario, la estrategia más sensata es:

1. No tocar la app `Streamlit` actual
2. Crear una versión nueva en paralelo
3. Reutilizar solo la lógica Python
4. Construir la nueva interfaz en `Next.js`
5. Desplegar frontend en `Vercel` y backend en `Render`

## Conclusión

Sí, existe una alternativa 100% gratuita con bastante más libertad de frontend que la solución actual.

La opción más equilibrada para tu caso es:

- `Next.js + Vercel Hobby`
- `FastAPI + Render Free`

Esto te da un frontend mucho más flexible sin afectar a la aplicación actual.
