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
