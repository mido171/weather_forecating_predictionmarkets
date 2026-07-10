PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)

.PHONY: bootstrap doctor-fast test-fast test-full validate lint format fetch-bootstrap archive-live experiment index milestones manifest clean

bootstrap:
	$(PYTHON) -m pip install -e ".[research,dev]"

doctor-fast:
	$(PYTHON) -m hkg_tmax doctor

test-fast:
	$(PYTHON) -m pytest tests/test_config_and_sources.py tests/test_experiments.py tests/test_hko_backfill.py tests/hkg_t24/test_h24n_contract_policy.py tests/hkg_t24/test_schema_sql_contract.py tests/test_demo_trading_migration.py

test-full:
	$(PYTHON) -m pytest tests

validate:
	$(PYTHON) -m hkg_tmax validate all

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

fetch-bootstrap:
	@test "$(EXECUTE)" = "1" || (echo 'Network disabled. Re-run with EXECUTE=1 after reviewing the source budget.' && exit 2)
	$(PYTHON) -m hkg_tmax sources fetch --tag bootstrap_now --max-sources 2 --execute

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
	@echo "Refusing an unbounded Makefile clean. Use the root repository cleaner in dry-run mode."
	@exit 2
