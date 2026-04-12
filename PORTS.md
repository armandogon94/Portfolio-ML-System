# Port Allocation — Project 07: Portfolio ML System

> All host-exposed ports are globally unique across all 16 projects so every project can run simultaneously. See `../PORT-MAP.md` for the full map.

## Current Assignments

| Service | Host Port | Container Port | File |
|---------|-----------|---------------|------|
| ML Dashboard (React) | **3070** | 3000 | docker-compose.yml / dev.yml |
| Prediction API (FastAPI) | **8070** | 8000 | docker-compose.yml / dev.yml |
| MLflow UI | **5070** | 5000 | docker-compose.yml / dev.yml |

## Allowed Range for New Services

If you need to add a new service to this project, pick from these ranges **only**:

| Type | Allowed Host Ports |
|------|--------------------|
| Frontend / UI | `3070 – 3079` |
| Backend / API | `8070 – 8079` |
| PostgreSQL | Not assigned. If needed, request an assignment in `../PORT-MAP.md`. |
| Redis | Not assigned. If needed, request an assignment in `../PORT-MAP.md`. |

Available slots: `3071-3079`, `8071-8079`.

## Do Not Use

Every port outside the ranges above is reserved by another project. Always check `../PORT-MAP.md` before picking a port.

Key ranges already taken:
- `3060-3069 / 8060-8069` → Project 06
- `3090-3099 / 8090-8099` → Project 09
- `5000` → macOS AirPlay — never use as a host port
- `5070` → MLflow (this project) — uses 5000 inside the container
- `6379-6385` → Projects 02, 05, 10, 12, 13, 15, 16 Redis
- `5432-5439` → Projects 02-05, 11-13, 15 PostgreSQL
