# Two stages: build the React bundle with Node, then run FastAPI, which serves
# both the API and that bundle. One image, one process, no separate web server.

FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first: this layer survives every change to the source.
COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN cd backend && uv sync --frozen --no-dev

COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist

# app/config.py resolves REPO_ROOT as parents[2] of backend/app/config.py,
# so the layout above has to mirror the repo: /app/backend, /app/frontend, /app/data.
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
