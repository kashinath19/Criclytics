# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn xgboost joblib matplotlib seaborn

# Expose port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "main2:app", "--host", "0.0.0.0", "--port", "8000"]
