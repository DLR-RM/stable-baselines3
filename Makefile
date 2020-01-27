SHELL=/bin/bash

pytest:
	./scripts/run_tests.sh

type:
	pytype

docs:
	cd docs && make html

spelling:
	cd docs && make spelling
