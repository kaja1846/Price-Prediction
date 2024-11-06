# Use Python as base image
FROM python

# Set working directory
WORKDIR /app

# Copy necessary files
COPY app.py /app
COPY model.joblib /app
COPY data/rental_1000.csv /app/data/rental_1000.csv

# Install dependencies
RUN pip install flask pandas joblib scikit-learn matplotlib

# Expose port for the Flask app
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]
