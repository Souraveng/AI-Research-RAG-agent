# Use a lightweight Python 3.10 image
FROM python:3.10-slim-bookworm

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy all your project files into the container
COPY . .

# Create a startup script to run BOTH the API and UI
RUN echo '#!/bin/bash\n\
# Start FastAPI in the background\n\
python main_api.py &\n\
# Start Streamlit in the foreground on port 7860\n\
streamlit run app_ui.py --server.port 7860 --server.address 0.0.0.0\n\
' > start.sh

# Make the script executable
RUN chmod +x start.sh

# Hugging Face Spaces expects apps to run on port 7860
EXPOSE 7860

# Command to run when the container starts
CMD ["./start.sh"]