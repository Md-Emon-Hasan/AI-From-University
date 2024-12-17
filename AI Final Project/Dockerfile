FROM python:3.11

# Install required packages
RUN pip install nltk

# Download nltk data (punkt)
RUN python -m nltk.downloader punkt

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]