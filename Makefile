SHELL=/bin/bash

pytest:
	./scripts/run_tests.sh

type:
	pytype

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean

.PHONY: clean spelling doc

# Build docker images
# If you do export RELEASE=True, it will also push them
docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# PyPi package release
release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/*

# Test PyPi package release
test-release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
