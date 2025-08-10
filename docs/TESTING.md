# Testing Guide for OAM 6G

This document provides an overview of the testing system for the OAM 6G project, including how to run tests, add new tests, and understand the test organization.

## Test Organization

The tests are organized into the following directories:

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for multiple components working together
- `tests/physics/`: Tests for physics simulation components
- `tests/regression/`: Tests to ensure that changes don't break existing functionality or degrade performance

## Running Tests

### Running All Tests

To run all tests, use the following command:

```bash
./run_tests.sh
```

### Running Specific Test Categories

To run specific test categories, use the `-t` or `--type` flag:

```bash
./run_tests.sh -t unit        # Run unit tests
./run_tests.sh -t integration # Run integration tests
./run_tests.sh -t physics     # Run physics tests
./run_tests.sh -t regression  # Run regression tests
./run_tests.sh -t smoke       # Run smoke tests (fast, basic functionality)
```

### Running Tests with Verbose Output

To run tests with verbose output, use the `-v` or `--verbose` flag:

```bash
./run_tests.sh -v
./run_tests.sh -t unit -v
```

### Running Tests with Coverage

To run tests with coverage reporting, use the following command:

```bash
./run_tests_with_coverage.sh
```

This will run all tests and generate an HTML coverage report in the `htmlcov/` directory. The report will be automatically opened in your default web browser.

## Adding New Tests

### Creating a New Test File

1. Create a new file in the appropriate test directory with a name starting with `test_`.
2. Import the necessary modules and components.
3. Create a test class with a name starting with `Test`.
4. Add test methods with names starting with `test_`.

Example:

```python
"""
Unit tests for the component.
"""

import pytest
import numpy as np
from my_module import MyComponent


class TestMyComponent:
    """Test the MyComponent class."""

    def test_initialization(self):
        """Test that the component initializes correctly."""
        component = MyComponent()
        assert component is not None
```

### Using Fixtures

Fixtures are defined in `tests/conftest.py` and can be used to set up test dependencies. To use a fixture, add it as a parameter to your test method:

```python
def test_with_fixture(self, base_config):
    """Test using the base_config fixture."""
    component = MyComponent(base_config)
    assert component is not None
```

### Using Markers

Markers are used to categorize tests. To add a marker to a test, use the `@pytest.mark` decorator:

```python
@pytest.mark.slow
def test_slow_operation(self):
    """Test a slow operation."""
    # ...
```

Available markers:
- `physics`: Tests for physics simulation components
- `integration`: Tests for integration between components
- `unit`: Tests for individual units/functions
- `regression`: Tests for regression testing
- `slow`: Tests that take a long time to run
- `fast`: Tests that run quickly
- `smoke`: Minimal subset of tests to verify basic functionality
- `parametrize`: Tests that use parametrization

### Parametrized Tests

Parametrized tests allow you to run the same test with different inputs:

```python
@pytest.mark.parametrize("input_value,expected_output", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(self, input_value, expected_output):
    """Test that the double function works correctly."""
    assert MyComponent.double(input_value) == expected_output
```

## Test Configuration

The test configuration is defined in the following files:

- `pytest.ini`: Configuration for pytest, including test discovery, markers, and output formatting
- `setup.cfg`: Configuration for coverage reporting and other tools
- `tests/conftest.py`: Fixtures and setup for pytest

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on the state of other tests.
2. **Test Coverage**: Aim for high test coverage, especially for critical components.
3. **Test Speed**: Tests should run as quickly as possible. Use the `@pytest.mark.slow` marker for tests that take a long time to run.
4. **Test Readability**: Tests should be easy to read and understand. Use descriptive test names and docstrings.
5. **Test Maintenance**: Keep tests up to date with code changes. Remove or update tests that are no longer relevant.
6. **Test Fixtures**: Use fixtures to set up test dependencies and avoid duplication.
7. **Test Assertions**: Use specific assertions to make test failures more informative.
8. **Test Organization**: Keep tests organized by category and component.
9. **Test Documentation**: Document test functions, classes, and modules.
10. **Test Consistency**: Follow consistent naming and organization conventions.

## Continuous Integration

The tests are run automatically on each pull request and push to the main branch using GitHub Actions. The configuration is defined in `.github/workflows/tests.yml`.

## Troubleshooting

If you encounter issues with running tests, try the following:

1. Make sure you have installed all the required dependencies:
   ```bash
   pip install -r config/requirements.txt
   pip install pytest pytest-cov
   ```

2. Make sure you are running the tests from the project root directory.

3. If a test is failing, try running it with verbose output to get more information:
   ```bash
   ./run_tests.sh -t unit -v
   ```

4. If a test is hanging or taking too long, try running it with a timeout:
   ```bash
   python -m pytest tests/path/to/test.py -v --timeout=30
   ```

5. If you're having issues with import errors, make sure the project root is in your Python path:
   ```python
   import sys
   import os
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
   ```