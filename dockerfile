FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY  ./src .
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
EXPOSE 8000
CMD ["python", "server.py"]
