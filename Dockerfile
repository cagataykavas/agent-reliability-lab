FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir .

ENTRYPOINT ["agent-reliability"]
CMD ["examples/cases.jsonl", "examples/traces.jsonl"]
