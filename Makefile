.PHONY: format lint run-all

# ruffでフォーマットを行う
format:
	rye run ruff format

# ruffでlintとフォーマットを行う
lint:
	rye run ruff check --fix