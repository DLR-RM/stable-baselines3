SHELL=/bin/bash

pytest:
	./scripts/run_tests.sh

pytype:
	pytype

doc:
	cd docs && make html

spelling:
	cd docs && make spelling
