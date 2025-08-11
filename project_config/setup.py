#!/usr/bin/env python3
"""
Setup script for OAM 6G - Orbital Angular Momentum Reinforcement Learning Environment

This package provides a complete RL environment for OAM-based wireless communication
with intelligent handover optimization using Deep Q-Networks.
"""

from setuptools import setup, find_packages
from pathlib import Path

                      
this_directory = Path(__file__).parent
long_description = (this_directory / "docs" / "README.md").read_text()

                   
requirements = (this_directory / "config" / "requirements.txt").read_text().splitlines()

setup(
    name="oam-6g",
    version="1.0.0",
    author="OAM 6G Research Team",
    author_email="research@oam6g.com",
    description="Orbital Angular Momentum Reinforcement Learning Environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oam6g/oam-rl-environment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Telecommunications",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
            "black==23.10.1",
            "flake8==6.1.0",
            "mypy==1.6.1",
        ],
        "docs": [
            "sphinx==7.2.6",
            "sphinx-rtd-theme==1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oam6g-train=scripts.main:main",
            "oam6g-evaluate=scripts.evaluation.evaluate_rl:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
) 
