all: install run

venv: .venv/bin/activate

.ONESHELL:
.venv/bin/activate:
	python -m venv .venv
	if [ "$(OS)" = "Windows_NT" ]; then \
		echo "Operating system is Windows"; \
		.venv/Scripts/Activate.ps1
	else \
		echo "Operating system is not Windows"; \
		source .venv/bin/activate
	fi
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
	pytest --tb=native

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
