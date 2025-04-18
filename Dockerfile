FROM python:3.12-slim

# Prevent Python from writing .pyc files to disk and enable output buffering.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt  
# Not using --no-cache-dir to not download the same packages again in the next build, which is useful for local development

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/main.py", "--server.enableCORS", "false"]
