# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py
PYTHON_VERSION = python3.9

# Install dependencies
install:
	$(PYTHON_VERSION) -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# Run the Flask application
run:
	FLASK_APP=$(FLASK_APP) ./$(VENV)/bin/flask run --debug --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install