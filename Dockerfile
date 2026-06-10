FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pageindex ./pageindex
COPY app ./app

ENV PAGEINDEX_DATA_DIR=/data
EXPOSE 8000

CMD ["python3", "app/server.py"]
