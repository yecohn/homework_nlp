#!/bin/bash

# Kill any existing processes on ports 8000 and 8501
kill_existing() {
    echo "Checking for existing processes..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:8501 | xargs kill -9 2>/dev/null
}

# Function to cleanup child processes on exit
cleanup() {
    echo "Shutting down services..."
    kill $(jobs -p)
    exit
}

# Set up trap to catch SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Kill existing processes
kill_existing

# Start FastAPI server
echo "Starting FastAPI server..."
python server.py &
# Wait a bit for the server to start
while ! curl -s http://localhost:8000 > /dev/null; do
    echo "Waiting for server to initialize..."
    sleep 10
done

# Start Streamlit
echo "Starting Streamlit app..."
streamlit run app.py &

# Wait for both processes
wait