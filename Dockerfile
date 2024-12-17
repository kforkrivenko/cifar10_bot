# Используем базовый образ Python
FROM python:3.12-slim

# Устанавливаем зависимости
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY . .

# Запускаем приложение
CMD ["python", "app.py"]
