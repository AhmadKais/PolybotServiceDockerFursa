# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make port 8443 available to the world outside this container
EXPOSE 8443

# Define environment variables
ENV TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
ENV TELEGRAM_APP_URL=${TELEGRAM_APP_URL}
ENV S3_BUCKET=${S3_BUCKET}
ENV YOLO_SERVICE_URL=${YOLO_SERVICE_URL}

# Run app.py when the container launches
CMD ["python", "app.py"]
