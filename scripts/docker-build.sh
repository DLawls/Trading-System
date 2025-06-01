#!/bin/bash

# Build the trading system Docker image
echo "Building trading system Docker image..."
docker build -t trading-system:latest .

echo "Build complete!"
echo "To run the container: docker-compose up"
echo "To run in development mode: docker-compose -f docker-compose.dev.yml up" 