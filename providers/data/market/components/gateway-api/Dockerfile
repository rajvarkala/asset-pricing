FROM python:3.12-slim

WORKDIR /app

COPY gateway-api/src /app/src

RUN pip install --no-cache-dir uv \
  && uv pip install --system \
       "fastapi>=0.115.12" \
       "uvicorn>=0.34.2" \
       "httpx>=0.28.1" \
       "pydantic-settings>=2.8.1"

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "gateway_api.main:app", "--host", "0.0.0.0", "--port", "8090", "--reload"]
