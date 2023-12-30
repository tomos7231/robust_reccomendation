.PHONY: env format
export PYTHONPATH := $(PWD)

env:
	pip install poetry
	poetry install

format:
	poetry run isort . & poetry run black .
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .