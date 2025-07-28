# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the full contents of the current context (project directory)
COPY . .

# Install the dependencies using the Tsinghua mirror
RUN pip install --no-cache-dir --no-build-isolation --break-system-packages -r requirements.txt || true



# Run the main script
CMD ["python3", "gnn2.py"]
