FROM python:3.9-slim

WORKDIR /app

COPY . .

CMD ["python", "test_heart_attack.py"]

USER nonrootuser
