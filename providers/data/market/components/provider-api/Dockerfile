FROM python:3.12-slim

WORKDIR /app

COPY db-interface/src /app/db-interface/src
COPY provider-api/src /app/src

RUN pip install --no-cache-dir uv \
  && uv pip install --system \
       "fastapi>=0.115.12" \
       "uvicorn>=0.34.2" \
       "sqlalchemy>=2.0.38" \
       "psycopg[binary]>=3.2.6" \
       "pydantic-settings>=2.8.1"

ENV PYTHONPATH=/app/src:/app/db-interface/src

CMD ["uvicorn", "provider_api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
