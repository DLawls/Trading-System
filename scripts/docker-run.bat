@echo off
echo Building and running trading system...

REM Build the image
docker build -t trading-system:latest .

REM Run with docker-compose
docker-compose up -d

echo Trading system is running!
echo Check logs with: docker-compose logs -f
echo Stop with: docker-compose down 