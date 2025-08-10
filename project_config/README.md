# Project Configuration Files

This directory contains project configuration files for testing, building, and development setup.

## Files

### Testing Configuration
- **`conftest.py`**: Pytest configuration for test discovery and path setup
- **`pytest.ini`**: Pytest settings including markers, test discovery, and output configuration
- **`run_tests.sh`**: Bash script for running different types of tests (unit, integration, physics, etc.)
- **`run_tests_with_coverage.sh`**: Script for running tests with coverage reporting

### Project Configuration
- **`pyproject.toml`**: Modern Python project configuration including build system, dependencies, and tool configurations
- **`setup.cfg`**: Legacy setup configuration (compatibility)
- **`setup.py`**: Package setup script for installation and distribution

## Usage

### Running Tests
```bash
# Run all tests
./project_config/run_tests.sh

# Run specific test types
./project_config/run_tests.sh -t unit
./project_config/run_tests.sh -t integration
./project_config/run_tests.sh -t physics

# Run with coverage
./project_config/run_tests_with_coverage.sh
```

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run with specific pytest options
python -m pytest tests/ -v --cov=.
```

## Configuration Details

### Test Categories
- **unit**: Individual component tests
- **integration**: Component interaction tests
- **physics**: Physics simulation validation tests
- **regression**: Regression testing
- **smoke**: Minimal functionality verification

### Coverage Reporting
- HTML coverage reports generated in `htmlcov/`
- Terminal coverage summary
- Coverage configuration in `setup.cfg` 