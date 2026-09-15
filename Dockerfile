FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps . && useradd --create-home evaluator && mkdir -p results && chown -R evaluator:evaluator /app
USER evaluator
ENTRYPOINT ["python", "-m", "llmguard.cli"]
CMD ["--help"]

