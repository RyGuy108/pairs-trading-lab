.PHONY: setup lint test build check demo walkthrough clean

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[all]"

lint:
	.venv/bin/python -m ruff check .

test:
	.venv/bin/python -m pytest

build:
	.venv/bin/python -m build

check: lint test build

demo:
	.venv/bin/pairs-trading demo

walkthrough:
	.venv/bin/python examples/walkthrough.py

clean:
	@echo "Generated outputs are intentionally preserved. Remove them manually if desired."
