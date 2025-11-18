# Use the official Python base image from Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the Flask app runs on (matching model.py)
EXPOSE 8080

# Define environment variable for Flask
ENV FLASK_APP=model.py

# Command to run the application when the container starts
CMD ["python", "model.py"]
