FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scheduler ./scheduler
COPY migrations ./migrations
COPY alembic.ini ./
COPY scripts/start.sh ./scripts/start.sh

EXPOSE 8000

ENTRYPOINT ["sh", "/app/scripts/start.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
