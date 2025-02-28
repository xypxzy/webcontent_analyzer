# WebContentAnalyzer Backend

Backend service for analyzing and optimizing web content.

## Features

- Page parsing and content extraction
- Text content analysis
- SEO recommendations generation
- HTML/CSS code optimization
- REST API for integration

## Technologies

- FastAPI
- PostgreSQL
- Celery with Redis
- SQLAlchemy
- Docker & Docker Compose

## Development Setup

### Prerequisites

- Docker and Docker Compose
- Make (optional, for using the Makefile commands)

### Quick Start

1. Clone the repository

```bash
git clone <repository-url>
cd backend
```

2. Start the services using Docker Compose

```bash
make up
# or without Makefile
docker-compose up -d
```

3. Run the database migrations

```bash
make migrate
# or without Makefile
docker-compose exec api alembic upgrade head
```

4. The API will be available at http://localhost:8000
5. API documentation is at http://localhost:8000/docs

### Using the Makefile

The project includes a Makefile to simplify common tasks:

```bash
# Start all services
make up

# Stop all services
make down

# View logs
make logs

# Open a shell in the API container
make shell

# Run migrations
make migrate

# Create a new migration
make alembic-revision

# Run tests
make test

# Run linting
make lint

# Format code with Black
make format
```

## Project Structure

```
backend/
├── app/                   # Application package
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   ├── crud/              # Database CRUD operations
│   ├── db/                # Database models and session
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   ├── services/          # Business logic services
│   │   ├── analyzer/      # Content analysis services
│   │   ├── generator/     # Code generation services
│   │   └── parser/        # Web page parsing services
│   └── tasks/             # Background tasks (Celery)
├── alembic/               # Database migrations
├── tests/                 # Test cases
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── .env.example           # Example environment variables
```

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

The following environment variables can be configured:

- `POSTGRES_SERVER`: PostgreSQL server hostname
- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DB`: PostgreSQL database name
- `POSTGRES_PORT`: PostgreSQL port
- `REDIS_HOST`: Redis hostname
- `REDIS_PORT`: Redis port
- `SECRET_KEY`: Secret key for JWT tokens
- `ENVIRONMENT`: Environment (development, production)
- `DEBUG`: Enable debug mode (true/false)
- `BACKEND_CORS_ORIGINS`: CORS origins list

## Running in Production

For production deployment, consider:

1. Using a proper secret key
2. Configuring CORS properly
3. Setting up SSL/TLS with a reverse proxy like Nginx
4. Using dedicated database and Redis instances
5. Setting up monitoring and logging