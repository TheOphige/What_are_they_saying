# Help target to list all commands
help:
	@echo "Available commands:"
	@awk '/^[a-zA-Z0-9_-]+:/{print $$1}' Makefile | sed 's/://'

# Install dependencies
install:
	pip install -r requirements.txt

# Run the Streamlit app
run:
	streamlit run main.py

# Run tests
test:
	pytest tests/

# Lint the code
lint:
	flake8 src/
