FROM python:3.12-slim

WORKDIR /app

COPY db-interface/src /app/src

RUN pip install --no-cache-dir uv \
  && uv pip install --system \
       "sqlalchemy>=2.0.38" \
       "psycopg[binary]>=3.2.6" \
       "pydantic-settings>=2.8.1"

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "db_interface.cli", "init-db"]
