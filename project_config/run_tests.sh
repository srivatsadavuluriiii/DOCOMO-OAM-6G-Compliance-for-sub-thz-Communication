#!/bin/bash

# Default values
TEST_TYPE="all"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -t|--type)
            TEST_TYPE="$2"
            shift
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-t|--type <test_type>] [-v|--verbose]"
            echo "  -t, --type      Test type to run (all, unit, integration, physics, regression, smoke)"
            echo "  -v, --verbose   Run tests in verbose mode"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set verbosity flag
if [ "$VERBOSE" = true ]; then
    VERBOSE_FLAG="-v"
else
    VERBOSE_FLAG=""
fi

# Run tests based on type
case $TEST_TYPE in
    all)
        echo "Running all tests..."
        python -m pytest $VERBOSE_FLAG tests/
        ;;
    unit)
        echo "Running unit tests..."
        python -m pytest $VERBOSE_FLAG tests/unit/
        ;;
    integration)
        echo "Running integration tests..."
        python -m pytest $VERBOSE_FLAG tests/integration/
        ;;
    physics)
        echo "Running physics tests..."
        python -m pytest $VERBOSE_FLAG tests/physics/
        ;;
    regression)
        echo "Running regression tests..."
        python -m pytest $VERBOSE_FLAG tests/regression/
        ;;
    smoke)
        echo "Running smoke tests..."
        python -m pytest $VERBOSE_FLAG -m smoke tests/
        ;;
    *)
        echo "Invalid test type: $TEST_TYPE"
        echo "Valid types: all, unit, integration, physics, regression, smoke"
        exit 1
        ;;
esac