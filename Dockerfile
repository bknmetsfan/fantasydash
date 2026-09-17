FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY blotter.py manual.json ./
ENV DATA_DIR=/data PORT=8080 TZ=America/New_York
EXPOSE 8080
# One worker: the state cache, sim and build lock are in-process.
CMD ["gunicorn", "-w", "1", "--threads", "8", "--timeout", "60", "-b", "0.0.0.0:8080", "blotter:app"]
