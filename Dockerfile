FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# Small committed copy of the 10 demo channels (scripts/make_demo_data.py); full data stays local.
COPY demo_data ./demo_data
COPY frontend ./frontend
ENV DATA_DIR=demo_data \
    CHANNELS=P-1,S-1,E-1,A-1,D-1,G-1,P-10,T-4,F-7,C-1
EXPOSE 8000
# Cloud hosts (Render, Railway, Fly) pass the port in $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
