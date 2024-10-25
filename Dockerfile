# Use the official TensorFlow image
FROM tensorflow/tensorflow:2.17.0

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Copy the model file from the project directory
COPY my_model.h5 ./my_model.h5

# Command to run your Flask app
CMD ["python", "MLOps.py"]
