clean:
	find . -type d -name "__pycache__" -print0 | xargs -0 rm -rf
	find . -name "wget-log" -print0 | xargs -0 rm -rf

# Create virtual environment if it does not exist
# set the venv and install requirements
install:
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		python -m venv .venv; \
	fi
	@echo "Installing requirements..."
	@.venv/bin/pip install -r requirements.txt
	@echo "Installation complete."

# set the venv if it does not exist do make install then run
run:
	@echo "Running the application..." 
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment not found. Installing..."; \
		make install; \
	fi
	@echo "Running the application..."
	export PYTHONPATH=$(shell pwd) && \
	. .venv/bin/activate && \
	streamlit run src/main.py