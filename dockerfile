FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./scripts /app/scripts
COPY ./Models /app/Models

EXPOSE 80

CMD ["python", "./scripts/external_infer.py"]
