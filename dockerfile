FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt update && apt install -y curl
RUN pip install --no-cache-dir -r requirements.txt
COPY  ./src .
RUN chmod +x ./run.sh
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
EXPOSE 8000
EXPOSE 8501
CMD ["./run.sh"]
