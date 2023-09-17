SHELL=/bin/bash
LINT_PATHS=stable_baselines3/ tests/ docs/conf.py

pytest:
	./scripts/run_tests.sh

pytype:
	pytype -j auto

mypy:
	mypy ${LINT_PATHS}

missing-annotations:
	mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports stable_baselines3

# missing docstrings
# pylint -d R,C,W,E -e C0116 stable_baselines3 -j 4

type: pytype mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	ruff --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean

# Build docker images
# If you do export RELEASE=True, it will also push them
docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# PyPi package release
release:
	poetry build
	poetry publish

# Test PyPi package release
test-release:
	poetry build
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry publish

.PHONY: clean spelling doc lint format check-codestyle commit-checks
