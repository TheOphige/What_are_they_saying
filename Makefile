# Install dependencies
install:
	pip install -r requirements.txt

# Run the Streamlit app
run:
	streamlit run streamlit_app/main.py

# Run tests
test:
	pytest tests/

# Lint the code
lint:
	flake8 src/
