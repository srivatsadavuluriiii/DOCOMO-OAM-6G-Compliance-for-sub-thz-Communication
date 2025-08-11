#!/usr/bin/env python3
"""
Centralized path management utilities.
"""

import os
import sys
from typing import Optional


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
                                                                     
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
                                                      
    while current_dir != os.path.dirname(current_dir):                           
                                                   
        if os.path.exists(os.path.join(current_dir, 'environment')) and\
           os.path.exists(os.path.join(current_dir, 'simulator')) and\
           os.path.exists(os.path.join(current_dir, 'models')):
            return current_dir
        
        current_dir = os.path.dirname(current_dir)
    
                                                               
    return os.getcwd()


def add_project_root_to_path() -> None:
    """
    Add the project root to sys.path.
    """
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def ensure_project_root_in_path() -> None:
    """
    Ensure the project root is in sys.path.
    """
    add_project_root_to_path()


def get_relative_path(relative_path: str) -> str:
    """
    Get an absolute path from a relative path within the project.
    
    Args:
        relative_path: Relative path from project root
        
    Returns:
        Absolute path
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


def find_file_in_project(filename: str) -> Optional[str]:
    """
    Find a file in the project directory tree.
    
    Args:
        filename: Name of the file to find
        
    Returns:
        Absolute path to the file if found, None otherwise
    """
    project_root = get_project_root()
    
    for root, dirs, files in os.walk(project_root):
        if filename in files:
            return os.path.join(root, filename)
    
    return None


def create_results_dir(name: str) -> str:
    """
    Create a results directory with the given name.
    
    Args:
        name: Name for the results directory
        
    Returns:
        Path to the created results directory
    """
    project_root = get_project_root()
    results_dir = os.path.join(project_root, 'results', name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir 
