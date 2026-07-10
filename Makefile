PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/code/src:$(CURDIR)

.PHONY: bootstrap doctor test validate lint format fetch-bootstrap archive-live experiment index milestones manifest clean

bootstrap:
	bash scripts/bootstrap.sh

doctor:
	$(PYTHON) -m hkg_tmax doctor

test:
	$(PYTHON) -m pytest

validate:
	$(PYTHON) -m hkg_tmax validate all

lint:
	$(PYTHON) -m ruff check code/src code/tests scripts
	$(PYTHON) -m mypy code/src

format:
	$(PYTHON) -m ruff format code/src code/tests scripts
	$(PYTHON) -m ruff check --fix code/src code/tests scripts

fetch-bootstrap:
	$(PYTHON) -m hkg_tmax sources fetch --tag bootstrap_now

archive-live:
	bash scripts/archive_live_loop.sh

experiment:
	@test -n "$(TITLE)" || (echo 'Usage: make experiment TITLE="Hypothesis title"' && exit 2)
	$(PYTHON) -m hkg_tmax experiments create --title "$(TITLE)"

index:
	$(PYTHON) -m hkg_tmax experiments index

milestones:
	$(PYTHON) -m hkg_tmax milestones render

manifest:
	$(PYTHON) -m hkg_tmax manifest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist
	find code/src code/tests -type d -name __pycache__ -prune -exec rm -rf {} +
