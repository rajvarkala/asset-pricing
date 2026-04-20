FROM python:3.12-slim

WORKDIR /app

COPY db-interface/src /app/db-interface/src
COPY instruments-crawler/src /app/src

RUN pip install --no-cache-dir uv \
  && uv pip install --system \
       "sqlalchemy>=2.0.38" \
       "psycopg[binary]>=3.2.6" \
       "pydantic-settings>=2.8.1" \
       "kiteconnect>=5.0.1"

ENV PYTHONPATH=/app/src:/app/db-interface/src

CMD ["python", "-m", "instruments_crawler.cli", "run-daemon"]
