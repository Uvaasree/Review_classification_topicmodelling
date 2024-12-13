FROM python:3.10-slim

WORKDIR /app

# Step 1: Copy only the requirement.txt
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Step 2: Now copy the rest of the files
COPY . .

EXPOSE 5001

CMD ["python", "main.py"]


# FROM python:3.9-slim: Uses a minimal Python image (Python 3.9) to keep the image lightweight.
# WORKDIR /app: Sets the working directory to /app inside the container.
# COPY . .: Copies the entire project directory (including requirements.txt) into the container.

# RUN pip install --no-cache-dir -r requirements.txt: Installs all dependencies listed in requirements.txt.
# EXPOSE 5001: Exposes port 5001 for the Flask app.
# CMD ["python", "main.py"]: Specifies the default command to run when the container starts.
