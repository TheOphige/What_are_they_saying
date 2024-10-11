# Help target to list all commands
help:
	@echo "Available commands:"
	@awk '/^[a-zA-Z0-9_-]+:/{print $$1}' Makefile | sed 's/://'

# Install dependencies
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# Run the Streamlit app
run:
	streamlit run main.py

# Run tests
test:
	pytest tests/

# format code
format:
	black *.py

# Lint the code
lint:
	flake8 src/


all: install lint test format