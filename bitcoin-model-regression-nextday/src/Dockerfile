# Use official Python image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (optimizes Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code AFTER installing dependencies
COPY . .

# Ensure the script has execution permissions (just in case)
RUN chmod +x main.py

# Run the model script
CMD ["python", "main.py"]
