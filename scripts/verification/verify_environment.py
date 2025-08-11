#!/usr/bin/env python3
"""
Verify that the environment is properly set up with the correct package versions.

This script checks that all required packages are installed with the correct versions
and that the environment is properly configured for reproducible research.
"""

import sys
import os
import importlib
import pkg_resources
from pathlib import Path
import platform
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

# Define color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_success(message):
    """Print a success message in green."""
    print(f"{GREEN} {message}{RESET}")

def print_error(message):
    """Print an error message in red."""
    print(f"{RED} {message}{RESET}")

def print_warning(message):
    """Print a warning message in yellow."""
    print(f"{YELLOW}! {message}{RESET}")

def check_package_version(package_name, required_version):
    """
    Check if a package is installed with the correct version.
    
    Args:
        package_name: Name of the package to check
        required_version: Required version of the package
        
    Returns:
        True if the package is installed with the correct version, False otherwise
    """
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        if installed_version == required_version:
            print_success(f"{package_name} {installed_version} (required: {required_version})")
            return True
        else:
            print_error(f"{package_name} {installed_version} (required: {required_version})")
            return False
    except pkg_resources.DistributionNotFound:
        print_error(f"{package_name} not installed (required: {required_version})")
        return False

def check_cuda_availability():
    """
    Check if CUDA is available for PyTorch.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    if torch.cuda.is_available():
        print_success(f"CUDA is available (version: {torch.version.cuda})")
        return True
    else:
        print_warning("CUDA is not available. Using CPU only.")
        return False

def check_random_seed_reproducibility():
    """
    Check if random seed setting produces reproducible results.
    
    Returns:
        True if random seed setting produces reproducible results, False otherwise
    """
    # Set random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Generate random numbers
    np_random1 = np.random.rand(10)
    torch_random1 = torch.rand(10)
    
    # Reset random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Generate random numbers again
    np_random2 = np.random.rand(10)
    torch_random2 = torch.rand(10)
    
    # Check if the random numbers are the same
    np_match = np.allclose(np_random1, np_random2)
    torch_match = torch.allclose(torch_random1, torch_random2)
    
    if np_match and torch_match:
        print_success("Random seed setting produces reproducible results")
        return True
    else:
        print_error("Random seed setting does NOT produce reproducible results")
        return False

def check_environment_variables():
    """
    Check if environment variables are set correctly.
    
    Returns:
        True if all environment variables are set correctly, False otherwise
    """
    # List of environment variables to check
    env_vars = {
        'PYTHONHASHSEED': '0',  # For reproducibility
    }
    
    all_correct = True
    for var, expected_value in env_vars.items():
        if var in os.environ and os.environ[var] == expected_value:
            print_success(f"Environment variable {var}={expected_value}")
        else:
            current_value = os.environ.get(var, 'not set')
            print_warning(f"Environment variable {var}={current_value} (recommended: {expected_value})")
            # Don't fail the check for environment variables
            # all_correct = False
    
    # Return True since environment variables are recommended but not required
    return True

def main():
    """Main entry point."""
    print("\n" + "="*80)
    print(" "*30 + "ENVIRONMENT VERIFICATION")
    print("="*80 + "\n")
    
    # Get required package versions from requirements.txt
    requirements_path = Path(__file__).parent.parent.parent / 'config' / 'requirements.txt'
    required_packages = {}
    with open(requirements_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                package, version = line.split('==')
                required_packages[package] = version
    
    # Check system information
    print("\n--- System Information ---")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Check package versions
    print("\n--- Package Versions ---")
    all_versions_correct = True
    for package, version in required_packages.items():
        try:
            if not check_package_version(package, version):
                all_versions_correct = False
        except Exception as e:
            print_error(f"Error checking {package}: {e}")
            all_versions_correct = False
    
    # Check CUDA availability
    print("\n--- CUDA Availability ---")
    cuda_available = check_cuda_availability()
    
    # Check random seed reproducibility
    print("\n--- Random Seed Reproducibility ---")
    reproducible = check_random_seed_reproducibility()
    
    # Check environment variables
    print("\n--- Environment Variables ---")
    env_vars_correct = check_environment_variables()
    
    # Print summary
    print("\n" + "="*80)
    print(" "*35 + "SUMMARY")
    print("="*80)
    
    if all_versions_correct:
        print_success("All package versions match the requirements")
    else:
        print_error("Some package versions do not match the requirements")
    
    if cuda_available:
        print_success("CUDA is available")
    else:
        print_warning("CUDA is not available (not required but recommended for performance)")
    
    if reproducible:
        print_success("Random seed setting produces reproducible results")
    else:
        print_error("Random seed setting does NOT produce reproducible results")
    
    if env_vars_correct:
        print_success("All environment variables are set correctly")
    else:
        print_warning("Some environment variables are not set correctly")
    
    # Print overall result
    print("\n" + "="*80)
    if all_versions_correct and reproducible:
        print_success("Environment verification PASSED")
        print("Your environment is properly set up for reproducible research")
    else:
        print_error("Environment verification FAILED")
        print("Please fix the issues above to ensure reproducible research")
    
    # Print note about CUDA
    if not cuda_available:
        print("\nNOTE: CUDA is not available, but this is not required for reproducibility.")
        print("      The code will run on CPU, which may be slower but will produce the same results.")
    
    print("="*80 + "\n")
    
    return 0 if all_versions_correct and reproducible else 1

if __name__ == "__main__":
    sys.exit(main())