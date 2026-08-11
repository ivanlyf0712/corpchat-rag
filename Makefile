# CorpChat RAG — Makefile
# ========================
# Operational helpers: validate .env, bootstrap external volumes, then run the
# compose stack (postgres + hindsight + corpchat-rag).

.PHONY: help check-env volumes up down logs ps test build

help:
	@echo "CorpChat RAG targets:"
	@echo "  make check-env   Validate .env has required secrets (DEEPSEEK_API_KEY, DB_PASSWORD)"
	@echo "  make volumes     Create external docker volumes if missing (ocr-platform_pgdata, hindsight-data)"
	@echo "  make up          check-env + volumes + docker compose up -d --build"
	@echo "  make down        docker compose down (keeps data volumes)"
	@echo "  make logs        Tail all service logs"
	@echo "  make ps          Show service status"
	@echo "  make test        Run the full pytest suite"
	@echo "  make build       docker compose build"

# ── .env validation ───────────────────────────────────────────────────
# DEEPSEEK_API_KEY is required by docker-compose (fail-fast); DB_PASSWORD is
# required by core.db.get_db_connection for local tooling (compose sets it).
check-env:
	@test -f .env || (echo "ERROR: .env not found — copy .env.example or create one with DEEPSEEK_API_KEY and DB_PASSWORD" && exit 1)
	@grep -q '^DEEPSEEK_API_KEY=.' .env || (echo "ERROR: DEEPSEEK_API_KEY is missing in .env (required by docker-compose)" && exit 1)
	@grep -q '^DB_PASSWORD=.' .env || (echo "WARNING: DB_PASSWORD is not in .env — compose passes its own, but local CLI/tooling needs it")
	@echo "✅ .env OK"

# ── External volumes (declared external:true in docker-compose.yml) ───
volumes:
	@docker volume inspect ocr-platform_pgdata >/dev/null 2>&1 || (docker volume create ocr-platform_pgdata && echo "created ocr-platform_pgdata")
	@docker volume inspect hindsight-data >/dev/null 2>&1 || (docker volume create hindsight-data && echo "created hindsight-data")
	@echo "✅ volumes ready"

up: check-env volumes
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

test:
	@/Users/ivanlee/miniconda3/envs/ocr/bin/python -m pytest tests/ -q

build:
	docker compose build
