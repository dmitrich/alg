"""
Tests for TensorBoard writer functionality.
Includes both unit tests and property-based tests for TensorBoardWriter class.
"""

import pytest
from hypothesis import given, strategies as st
import os
import tempfile
import shutil
import sys
from unittest.mock import patch, MagicMock
from utils_tensorboard.writer import TensorBoardWriter


class TestTensorBoardWriterUnit:
    """Unit tests for TensorBoardWriter class."""
    
    def test_initialization_success(self):
        """Test successful TensorBoard writer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            
            # Check that log_dir is set correctly
            assert writer.log_dir == log_dir
            
            # If TensorBoard is available, check writer is initialized
            if writer.is_available():
                assert writer.writer is not None
                writer.close()
            else:
                # If not available, writer should be None
                assert writer.writer is None
    
    def test_initialization_import_failure(self):
        """Test graceful handling of TensorBoard import failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock import failure by patching the import statement
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if 'torch.utils.tensorboard' in name:
                    raise ImportError("Mocked import failure")
                return real_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with patch('builtins.print') as mock_print:
                    writer = TensorBoardWriter(log_dir)
                    
                    # Check that writer is not available
                    assert not writer.is_available()
                    assert writer.writer is None
                    
                    # Check warning was printed
                    assert mock_print.call_count >= 1
                    print_output = ' '.join([str(call) for call in mock_print.call_args_list])
                    assert "Warning: TensorBoard not available" in print_output
    
    def test_initialization_general_exception(self):
        """Test graceful handling of general initialization exceptions."""
        # Skip this test if TensorBoard is not installed
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            pytest.skip("TensorBoard not installed, skipping exception handling test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock SummaryWriter to raise an exception
            mock_summary_writer = MagicMock(side_effect=RuntimeError("Test error"))
            
            with patch('torch.utils.tensorboard.SummaryWriter', mock_summary_writer):
                with patch('builtins.print') as mock_print:
                    writer = TensorBoardWriter(log_dir)
                    
                    # Check that writer is not available
                    assert not writer.is_available()
                    assert writer.writer is None
                    
                    # Check warning was printed with error message
                    assert mock_print.call_count >= 1
                    print_output = ' '.join([str(call) for call in mock_print.call_args_list])
                    assert "Warning: TensorBoard initialization failed" in print_output
                    assert "Test error" in print_output
    
    def test_close_prints_instructions(self):
        """Test that close() prints TensorBoard log location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            
            if writer.is_available():
                with patch('builtins.print') as mock_print:
                    writer.close()
                    
                    # Check that log location was printed
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    all_output = ' '.join(print_calls)
                    
                    assert log_dir in all_output
                    assert 'TensorBoard logs saved' in all_output
    
    def test_close_when_unavailable(self):
        """Test that close() does nothing when TensorBoard is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock import failure
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if 'torch.utils.tensorboard' in name:
                    raise ImportError("Mocked import failure")
                return real_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with patch('builtins.print'):
                    writer = TensorBoardWriter(log_dir)
                
                # close() should not raise an exception
                with patch('builtins.print') as mock_print:
                    writer.close()
                    
                    # Should not print instructions when unavailable
                    assert mock_print.call_count == 0
    
    def test_is_available_returns_correct_status(self):
        """Test that is_available() returns correct status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Test with current environment (may or may not have TensorBoard)
            writer = TensorBoardWriter(log_dir)
            assert writer.is_available() == writer.available
            if writer.is_available():
                writer.close()
            
            # Test with failed initialization
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if 'torch.utils.tensorboard' in name:
                    raise ImportError("Mocked import failure")
                return real_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with patch('builtins.print'):
                    writer = TensorBoardWriter(log_dir)
                    assert not writer.is_available()
                    assert writer.is_available() == writer.available
    
    def test_dual_location_logging(self):
        """Test that logs are copied to secondary location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            secondary_log_dir = os.path.join(tmpdir, 'run_root')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir, secondary_log_dir)
            
            if writer.is_available():
                # Close writer to trigger file copy
                writer.close()
                
                # Check that event files exist in both locations
                primary_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
                secondary_files = [f for f in os.listdir(secondary_log_dir) if f.startswith('events.out.tfevents')]
                
                # Both locations should have event files
                assert len(primary_files) > 0
                assert len(secondary_files) > 0
                
                # Files should be the same
                assert set(primary_files) == set(secondary_files)
    
    def test_dual_location_logging_handles_copy_failure(self):
        """Test graceful handling when secondary location copy fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            secondary_log_dir = '/invalid/path/that/does/not/exist'
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir, secondary_log_dir)
            
            if writer.is_available():
                with patch('builtins.print') as mock_print:
                    # Should not raise an exception
                    writer.close()
                    
                    # Should print warning about copy failure
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    all_output = ' '.join(print_calls)
                    assert 'Warning' in all_output or 'Failed to copy' in all_output


class TestTensorBoardWriterProperties:
    """Property-based tests for TensorBoardWriter class."""
    
    # Feature: tensorboard-integration, Property 3: Directory Structure Compliance
    @given(
        run_id=st.text(
            alphabet=st.characters(whitelist_categories=('Ll', 'Nd'), whitelist_characters='-_'),
            min_size=5,
            max_size=30
        ).filter(lambda x: x and not x[0].isdigit())
    )
    def test_property_3_directory_structure_compliance(self, run_id):
        """
        Property 3: Directory Structure Compliance
        
        For any training run with run_id R, TensorBoard event files SHALL be
        created in the path runs/R/logs/tensorboard/ and nowhere else.
        
        **Validates: Requirements 2.1, 2.3**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, 'runs')
            log_dir = os.path.join(runs_dir, run_id, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            
            # Verify the log_dir is set correctly
            assert writer.log_dir == log_dir
            
            # Verify the path structure
            assert 'runs' in log_dir
            assert run_id in log_dir
            assert log_dir.endswith(os.path.join('logs', 'tensorboard'))
            
            # If TensorBoard is available, verify files are created in correct location
            if writer.is_available():
                writer.close()
                
                # Check that event files exist in the correct directory
                files = os.listdir(log_dir)
                event_files = [f for f in files if f.startswith('events.out.tfevents')]
                
                # At least one event file should be created
                assert len(event_files) > 0
                
                # Verify no event files exist outside this directory
                for root, dirs, files in os.walk(runs_dir):
                    if root != log_dir:
                        event_files_outside = [f for f in files if f.startswith('events.out.tfevents')]
                        assert len(event_files_outside) == 0, f"Found event files outside log_dir: {root}"
    
    @given(
        path_components=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=('Ll', 'Nd'), whitelist_characters='-_'),
                min_size=1,
                max_size=20
            ).filter(lambda x: x and not x[0].isdigit()),
            min_size=1,
            max_size=5
        )
    )
    def test_writer_handles_various_paths(self, path_components):
        """
        Test that TensorBoardWriter handles various valid directory paths.
        
        For any valid directory path, the writer should initialize correctly
        and store the path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, *path_components)
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            
            # Verify log_dir is stored correctly
            assert writer.log_dir == log_dir
            
            # Verify writer state is consistent
            assert isinstance(writer.available, bool)
            assert writer.is_available() == writer.available
            
            # Clean up
            if writer.is_available():
                writer.close()
    
    def test_multiple_writers_isolated(self):
        """
        Test that multiple TensorBoardWriter instances are isolated.
        
        Creating multiple writers with different log directories should
        not interfere with each other.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir1 = os.path.join(tmpdir, 'run1', 'logs', 'tensorboard')
            log_dir2 = os.path.join(tmpdir, 'run2', 'logs', 'tensorboard')
            os.makedirs(log_dir1, exist_ok=True)
            os.makedirs(log_dir2, exist_ok=True)
            
            writer1 = TensorBoardWriter(log_dir1)
            writer2 = TensorBoardWriter(log_dir2)
            
            # Verify both writers are independent
            assert writer1.log_dir != writer2.log_dir
            assert writer1.log_dir == log_dir1
            assert writer2.log_dir == log_dir2
            
            # Both should have same availability status
            assert writer1.is_available() == writer2.is_available()
            
            # Clean up
            if writer1.is_available():
                writer1.close()
            if writer2.is_available():
                writer2.close()
            
            # Verify files are in separate directories
            if writer1.is_available():
                files1 = os.listdir(log_dir1)
                files2 = os.listdir(log_dir2)
                
                event_files1 = [f for f in files1 if f.startswith('events.out.tfevents')]
                event_files2 = [f for f in files2 if f.startswith('events.out.tfevents')]
                
                assert len(event_files1) > 0
                assert len(event_files2) > 0
                
                # Files should be different
                assert set(event_files1).isdisjoint(set(event_files2))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
