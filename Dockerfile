# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY bike_rental_api /app

RUN pip install --no-cache-dir -r requirements.txt && pip install *.whl

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Define environment variable to run the app using uvicorn
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8001

CMD ["python", "app/main.py"]