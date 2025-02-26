from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.parsing_tasks",
        "app.tasks.analysis_tasks",
    ],
)

celery_app.conf.task_routes = {
    "app.tasks.parsing_tasks.*": {"queue": "parsing"},
    "app.tasks.analysis_tasks.*": {"queue": "analysis"},
}

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)
