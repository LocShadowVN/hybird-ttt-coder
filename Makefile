.PHONY: all fmt lint rust-test build py-lint py-test eval clean

all: rust-test build py-lint py-test eval

fmt:
	cargo fmt --all
	ruff format .

lint:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings
	ruff check .
	ruff format --check .
	mypy model runtime train eval tests typings

rust-test:
	PYO3_PYTHON=$$(which python3) cargo test --all-targets

build:
	maturin develop --release

py-lint:
	ruff check .
	ruff format --check .
	mypy model runtime train eval tests typings

py-test:
	pytest --cov=model --cov=runtime --cov=train --cov=eval --cov-report=term-missing --cov-fail-under=80

eval:
	python3 -m eval.benchmark --preset tiny --output eval_report.json --markdown-out eval_summary.md --strict-ci

clean:
	cargo clean
	rm -rf dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache eval_report.json eval_summary.md
