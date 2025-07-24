# Lint
typing:
	uv run --isolated --only-group dev mypy --install-types --non-interactive --explicit-package-bases .

lint:
	uv run --isolated --only-group dev ruff check
	uv run --isolated --only-group dev black --check --diff .

fmt:
	uv run --isolated --only-group dev black .
	uv run --isolated --only-group dev ruff check --fix --unsafe-fixes
	$(MAKE) lint

.PHONY: typing fmt lint