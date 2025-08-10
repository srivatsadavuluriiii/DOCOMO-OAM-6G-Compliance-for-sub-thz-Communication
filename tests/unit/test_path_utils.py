"""
Unit tests for path utilities.
"""

import os
import sys
import pytest
from utils.path_utils import (
    get_project_root,
    add_project_root_to_path,
    ensure_project_root_in_path,
    get_relative_path,
    find_file_in_project
)


class TestPathUtils:
    """Test the path utilities."""

    def test_get_project_root(self):
        """Test that get_project_root returns a valid directory."""
        project_root = get_project_root()
        
        # Check that the project root is a string
        assert isinstance(project_root, str)
        
        # Check that the project root exists
        assert os.path.exists(project_root)
        
        # Check that the project root has the expected subdirectories
        assert os.path.exists(os.path.join(project_root, 'environment'))
        assert os.path.exists(os.path.join(project_root, 'simulator'))
        assert os.path.exists(os.path.join(project_root, 'models'))
    
    def test_add_project_root_to_path(self):
        """Test that add_project_root_to_path adds the project root to sys.path."""
        # Get the project root
        project_root = get_project_root()
        
        # Save the original sys.path
        original_sys_path = sys.path.copy()
        
        try:
            # Add the project root to sys.path
            add_project_root_to_path()
            
            # Check that the project root is in sys.path
            assert project_root in sys.path
            
            # Check that the project root is at the beginning of sys.path if it was added
            if sys.path[0] == project_root:
                # Project root was added at the beginning
                pass
            else:
                # Project root was already in sys.path
                assert project_root in sys.path
        finally:
            # Restore the original sys.path
            sys.path = original_sys_path
    
    def test_ensure_project_root_in_path(self):
        """Test that ensure_project_root_in_path ensures the project root is in sys.path."""
        # Get the project root
        project_root = get_project_root()
        
        # Save the original sys.path
        original_sys_path = sys.path.copy()
        
        try:
            # Ensure the project root is in sys.path
            ensure_project_root_in_path()
            
            # Check that the project root is in sys.path
            assert project_root in sys.path
            
            # Get the current count of the project root in sys.path
            initial_count = sys.path.count(project_root)
            
            # Call ensure_project_root_in_path again to check that it doesn't add duplicates
            ensure_project_root_in_path()
            
            # Count how many times the project root appears in sys.path
            new_count = sys.path.count(project_root)
            
            # Check that the count hasn't increased (no duplicates added)
            assert new_count == initial_count
        finally:
            # Restore the original sys.path
            sys.path = original_sys_path
    
    def test_get_relative_path(self):
        """Test that get_relative_path returns the correct absolute path."""
        # Get the project root
        project_root = get_project_root()
        
        # Get the absolute path to a relative path
        relative_path = 'environment/oam_env.py'
        absolute_path = get_relative_path(relative_path)
        
        # Check that the absolute path is correct
        assert absolute_path == os.path.join(project_root, relative_path)
        
        # Check that the absolute path exists
        assert os.path.exists(absolute_path)
    
    def test_find_file_in_project(self):
        """Test that find_file_in_project finds a file in the project."""
        # Find a file that should exist
        filename = 'oam_env.py'
        file_path = find_file_in_project(filename)
        
        # Check that the file was found
        assert file_path is not None
        
        # Check that the file exists
        assert os.path.exists(file_path)
        
        # Check that the file has the correct name
        assert os.path.basename(file_path) == filename
        
        # Find a file that should not exist
        filename = 'nonexistent_file.xyz'
        file_path = find_file_in_project(filename)
        
        # Check that the file was not found
        assert file_path is None