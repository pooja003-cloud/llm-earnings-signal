.PHONY: install test demo data score backtest report all

install:
	pip install -e ".[dev]"

test:
	pytest -q

demo:            ## offline run on synthetic data (no keys needed)
	earnsig demo

data:            ## prices, factors and SEC 8-K earnings releases
	earnsig prices
	earnsig factors
	earnsig collect

score:           ## LLM scores, consistency check, dictionary baseline
	earnsig score
	earnsig consistency
	earnsig baseline

backtest:
	earnsig backtest

report:          ## write results into README.md
	earnsig report

all: data score backtest report
