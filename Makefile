all: install run

venv: .venv/bin/activate

.ONESHELL:
.venv/bin/activate:
	python -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip

.PHONY:
dependencies_test: venv
	pip install .[test]

.PHONY:
dependencies_lint: venv
	pip install .[lint]

.PHONY:
test: venv
	source .venv/bin/activate
	pytest

.PHONY:
pre-commit: venv
	source .venv/bin/activate
	pre-commit autoupdate
	pre-commit install
	pre-commit run --all-files

.PHONY:
pre-commit-ci: venv
	source .venv/bin/activate
	SKIP=no-commit-to-branch pre-commit run --all-files
