#!/bin/bash

# Run tests with coverage
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term --cov-config=setup.cfg

# Open coverage report
if [[ "$OSTYPE" == "darwin"* ]]; then
    open htmlcov/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open htmlcov/index.html
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    start htmlcov/index.html
fi