FROM python:3.13-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install  -r requirements.txt

# Copy the rest of your application
COPY . .

# Create data directories if they don't exist
RUN mkdir -p /app/src/data

# Fix the InMemoryChatMessageHistory import issue
# RUN sed -i 's/from langchain_core.chat_history import InMemoryChatMessageHistory/from langchain.memory import ChatMessageHistory as InMemoryChatMessageHistory/g' /app/src/workflow.py


# Expose port 5000
EXPOSE 5000

# Run the Flask application
CMD ["python", "-m", "src.api.endpoints"]