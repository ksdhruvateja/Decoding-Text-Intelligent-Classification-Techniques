#!/bin/bash

# Text Classifier Quick Start Script
# This script sets up and runs the application

echo "=========================================="
echo "   Text Classifier - Quick Start Setup   "
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
echo "Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check if Node is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js 14 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"
echo -e "${GREEN}✓ Node found: $(node --version)${NC}"
echo ""

# Setup Backend
echo "=========================================="
echo "Setting up Backend..."
echo "=========================================="
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p checkpoints
mkdir -p data

# Generate demo data
echo "Generating demo data..."
python generate_demo_data.py

echo -e "${GREEN}✓ Backend setup complete!${NC}"
echo ""

# Setup Frontend
cd ../frontend
echo "=========================================="
echo "Setting up Frontend..."
echo "=========================================="

# Install npm dependencies
echo "Installing npm dependencies..."
npm install

echo -e "${GREEN}✓ Frontend setup complete!${NC}"
echo ""

# Final instructions
echo "=========================================="
echo "   Setup Complete! 🎉                    "
echo "=========================================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "The app will open at: http://localhost:3000"
echo "API will be running at: http://localhost:5000"
echo ""
echo "=========================================="
echo ""

# Ask if user wants to start now
read -p "Do you want to start the backend now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ../backend
    source venv/bin/activate
    echo "Starting backend server..."
    python app.py
fi