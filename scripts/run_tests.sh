#!/bin/bash
python3 -m pytest --cov-report html --cov-report term --cov=. -v --color=yes -m "not expensive" --durations=15
