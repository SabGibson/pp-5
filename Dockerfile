# Use an official Python runtime as a parent image
FROM python:3.10.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app


RUN pip install --no-cache-dir pipenv && pipenv install --system --deploy


EXPOSE 8501


CMD streamlit run app.py
