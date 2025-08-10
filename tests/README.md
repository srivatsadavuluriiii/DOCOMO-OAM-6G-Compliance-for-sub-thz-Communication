# OAM 6G Test Suite

This directory contains all tests for the OAM 6G project, organized by type and purpose.

## Directory Structure

### `/debug/`
- **Purpose**: Debug scripts and troubleshooting tools
- **Files**: 
  - `debug_position.py` - Debug position generation logic

### `/unit/`
- **Purpose**: Unit tests for individual components
- **Files**:
  - `test_distance_categorization.py` - Test distance optimization logic

### `/integration/`
- **Purpose**: Integration tests for system components
- **Files**:
  - `test_distance_performance.py` - Test distance optimization performance

## Running Tests

### Debug Tests
```bash
python tests/debug/debug_position.py
```

### Unit Tests
```bash
python tests/unit/test_distance_categorization.py
```

### Integration Tests
```bash
python tests/integration/test_distance_performance.py
```

### All Tests
```bash
python -m pytest tests/
```

## Test Guidelines

1. **Debug Tests**: Use for troubleshooting specific issues
2. **Unit Tests**: Test individual functions/classes in isolation
3. **Integration Tests**: Test how components work together

## Coverage

Run coverage analysis:
```bash
coverage run -m pytest tests/
coverage report
coverage html
``` 