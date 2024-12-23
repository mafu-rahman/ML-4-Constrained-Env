# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all application files to the container
COPY app_original.py ./app_original.py
COPY app_pca500.py ./app_pca500.py
COPY app_pca200.py ./app_pca200.py

# Copy models folder
COPY models ./models

# Copy requirements.txt and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default application port
EXPOSE 8000

# Use environment variables to determine which app to run
ARG APP_FILE=app_original
CMD ["sh", "-c", "uvicorn ${APP_FILE}:app --host 0.0.0.0 --port 8000"]