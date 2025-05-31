# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
# May need build-essential or other OS libraries for some Python packages (e.g. ones with C extensions)
# RUN apt-get update && apt-get install -y build-essential gcc
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local directory contents into the container at /app
COPY . .

# Make port 11434 available to the world outside this container (if Ollama ran inside, which it doesn't here)
# This is more for if the bot itself exposed an API. Ollama is expected to be external.
# EXPOSE 11434

# Define environment variables if needed (e.g. for config, though current script uses config.json)
# ENV NAME World

# Run trading_cli.py when the container launches
# The default command will be to show help. User can override with `docker run <image> --run-trader`
CMD ["python", "trading_cli.py", "--help"]
