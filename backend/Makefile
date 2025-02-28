.PHONY: up down logs shell migrate create-migration test lint format

# Run all services
up:
	docker-compose up -d

# Run all services in development mode with hot reloading
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Open a shell in the backend container
shell:
	docker-compose exec backend bash

# Run migrations
migrate:
	docker-compose exec backend alembic upgrade head

# Create a new migration
create-migration:
	@read -p "Enter migration message: " message; \
	docker-compose exec backend alembic revision --autogenerate -m "$$message"

# Run tests
test:
	docker-compose exec backend pytest

# Run linting
lint:
	docker-compose exec backend flake8 app

# Format code with Black
format:
	docker-compose exec backend black app

# Initialize database
init-db:
	docker-compose exec backend python -c "from app.db.init_db import init_db; import asyncio; asyncio.run(init_db())"

# View API documentation in browser
docs:
	@echo "API documentation available at: http://localhost:8000/docs"
	@echo "ReDoc documentation available at: http://localhost:8000/redoc"