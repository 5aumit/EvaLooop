FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

# Install only API, evaluation, and analysis dependencies. No GPU runtime.
COPY constraints-hw4.txt ./
RUN pip install --no-cache-dir -c constraints-hw4.txt openai python-dotenv fire evalplus==0.3.1 pandas matplotlib seaborn pyyaml
COPY pyproject.toml README.md ./
COPY evaloop ./evaloop
RUN pip install --no-deps .
COPY run_hw4.py analyze_hw4.py ./
RUN python run_hw4.py --check

ENTRYPOINT ["python", "/app/run_hw4.py"]
