FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY opsarena /app/opsarena
COPY server /app/server
COPY baselines /app/baselines
COPY scripts /app/scripts
COPY configs /app/configs
COPY data /app/data

COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml
COPY __init__.py /app/__init__.py
COPY client.py /app/client.py
COPY models.py /app/models.py

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
