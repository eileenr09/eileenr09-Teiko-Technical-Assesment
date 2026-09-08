.PHONY: setup pipeline dashboard

PYTHON := $(shell command -v python3 2>/dev/null || command -v python)

setup:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) load_data.py
	$(PYTHON) analysis.py

dashboard:
	$(PYTHON) -m streamlit run dashboard.py --server.headless true
