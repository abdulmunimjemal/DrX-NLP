clean:
	find . -type d -name "__pycache__" -print0 | xargs -0 rm -rf
	find . -name "wget-log" -print0 | xargs -0 rm -rf

run:
	export PYTHONPATH=$(shell pwd) && \
	streamlit run src/main.py