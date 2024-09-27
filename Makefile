# Makefile for KMeans Clustering Web Application

# Define variables for Python and Flask
PYTHON := python3
PIP := pip3
FLASK := flask
APP := app.py

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Run the web application locally on http://localhost:3000
run:
	$(FLASK) run --host=localhost --port=3000

# Clean up any __pycache__ or temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

.PHONY: install run clean

