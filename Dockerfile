FROM python:3.13.18-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PAGEINDEX_DATA_DIR=/data
# Caps the flash parser pool (os.cpu_count() - 1 processes) so a parse fits mem_limit.
ENV PYTHON_CPU_COUNT=4
EXPOSE 8000

CMD ["python3", "app/server.py"]
