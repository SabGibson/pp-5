# Start from the official Python image
FROM python:3.10-slim-buster

# Create a working directory
WORKDIR /app

# Copy the requirements file in order to install 
# Python dependencies
COPY requirements.txt ./requirements.txt

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port for the Streamlit application
ENV PORT 8501
EXPOSE $PORT

# Run the application
CMD streamlit run --server.address 0.0.0.0 --server.port $PORT app.py

