# Use an official Python runtime as the base image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt ./


    
# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code to the working directory
COPY . .

# Set build arguments
ARG GOOGLE_API_KEY='my_build key'

# Set environment variables using build arguments
ENV GOOGLE_API_KEY=$GOOGLE_API_KEY

EXPOSE 8501

CMD ["streamlit","run","app.py"]