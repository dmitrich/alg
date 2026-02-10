"""
Tests for TensorBoard metric logger functionality.
Includes both unit tests and property-based tests for MetricLogger class.
"""

import pytest
from hypothesis import given, strategies as st
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from utils_tensorboard.writer import TensorBoardWriter
from utils_tensorboard.logger import MetricLogger


class TestMetricLoggerUnit:
    """Unit tests for MetricLogger class."""
    
    def test_initialization(self):
        """Test MetricLogger initialization with TensorBoardWriter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            # Check that writer is stored correctly
            assert logger.writer is writer
            
            if writer.is_available():
                writer.close()
    
    def test_log_training_loss_success(self):
        """Test successful logging of training loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock the add_scalar method
                writer.writer.add_scalar = MagicMock()
                
                # Log training loss
                logger.log_training_loss(0.5, 100)
                
                # Verify add_scalar was called with correct arguments
                writer.writer.add_scalar.assert_called_once_with('Loss/train_step', 0.5, 100)
                
                writer.close()
    
    def test_log_training_loss_when_unavailable(self):
        """Test that logging does nothing when TensorBoard is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Create writer with mocked import failure
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if 'torch.utils.tensorboard' in name:
                    raise ImportError("Mocked import failure")
                return real_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with patch('builtins.print'):
                    writer = TensorBoardWriter(log_dir)
            
            logger = MetricLogger(writer)
            
            # This should not raise an exception
            logger.log_training_loss(0.5, 100)
    
    def test_log_training_loss_handles_exception(self):
        """Test graceful handling of exceptions during training loss logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock add_scalar to raise an exception
                writer.writer.add_scalar = MagicMock(side_effect=RuntimeError("Test error"))
                
                with patch('builtins.print') as mock_print:
                    # This should not raise an exception
                    logger.log_training_loss(0.5, 100)
                    
                    # Verify warning was printed
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    all_output = ' '.join(print_calls)
                    assert "Warning: Failed to log training loss" in all_output
                    assert "Test error" in all_output
                
                # Verify writer is marked as unavailable
                assert not writer.is_available()
                
                writer.close()
    
    def test_log_evaluation_metrics_success(self):
        """Test successful logging of evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock the add_scalar method
                writer.writer.add_scalar = MagicMock()
                
                # Log evaluation metrics
                logger.log_evaluation_metrics(0.4, 0.6, 0.001, 500)
                
                # Verify add_scalar was called three times with correct arguments
                assert writer.writer.add_scalar.call_count == 3
                calls = writer.writer.add_scalar.call_args_list
                
                # Check each call
                assert call('Loss/train', 0.4, 500) in calls
                assert call('Loss/val', 0.6, 500) in calls
                assert call('Learning_Rate', 0.001, 500) in calls
                
                writer.close()
    
    def test_log_evaluation_metrics_when_unavailable(self):
        """Test that evaluation metrics logging does nothing when TensorBoard is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Create writer with mocked import failure
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if 'torch.utils.tensorboard' in name:
                    raise ImportError("Mocked import failure")
                return real_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with patch('builtins.print'):
                    writer = TensorBoardWriter(log_dir)
            
            logger = MetricLogger(writer)
            
            # This should not raise an exception
            logger.log_evaluation_metrics(0.4, 0.6, 0.001, 500)
    
    def test_log_evaluation_metrics_handles_exception(self):
        """Test graceful handling of exceptions during evaluation metrics logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock add_scalar to raise an exception
                writer.writer.add_scalar = MagicMock(side_effect=RuntimeError("Test error"))
                
                with patch('builtins.print') as mock_print:
                    # This should not raise an exception
                    logger.log_evaluation_metrics(0.4, 0.6, 0.001, 500)
                    
                    # Verify warning was printed
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    all_output = ' '.join(print_calls)
                    assert "Warning: Failed to log evaluation metrics" in all_output
                    assert "Test error" in all_output
                
                # Verify writer is marked as unavailable
                assert not writer.is_available()
                
                writer.close()
    
    def test_log_gradient_norm_success(self):
        """Test successful logging of gradient norm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock the add_scalar method
                writer.writer.add_scalar = MagicMock()
                
                # Log gradient norm
                logger.log_gradient_norm(1.5, 200)
                
                # Verify add_scalar was called with correct arguments
                writer.writer.add_scalar.assert_called_once_with('Gradients/norm', 1.5, 200)
                
                writer.close()
    
    def test_log_gradient_norm_when_unavailable(self):
        """Test that gradient norm logging does nothing when TensorBoard is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Create writer with mocked import failure
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if 'torch.utils.tensorboard' in name:
                    raise ImportError("Mocked import failure")
                return real_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with patch('builtins.print'):
                    writer = TensorBoardWriter(log_dir)
            
            logger = MetricLogger(writer)
            
            # This should not raise an exception
            logger.log_gradient_norm(1.5, 200)
    
    def test_log_gradient_norm_handles_exception(self):
        """Test graceful handling of exceptions during gradient norm logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock add_scalar to raise an exception
                writer.writer.add_scalar = MagicMock(side_effect=RuntimeError("Test error"))
                
                with patch('builtins.print') as mock_print:
                    # This should not raise an exception
                    logger.log_gradient_norm(1.5, 200)
                    
                    # Verify warning was printed
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    all_output = ' '.join(print_calls)
                    assert "Warning: Failed to log gradient norm" in all_output
                    assert "Test error" in all_output
                
                # Verify writer is marked as unavailable
                assert not writer.is_available()
                
                writer.close()
    
    def test_multiple_logging_calls(self):
        """Test multiple logging calls in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock the add_scalar method
                writer.writer.add_scalar = MagicMock()
                
                # Log multiple metrics
                logger.log_training_loss(0.5, 100)
                logger.log_training_loss(0.4, 101)
                logger.log_evaluation_metrics(0.4, 0.6, 0.001, 100)
                logger.log_gradient_norm(1.5, 100)
                
                # Verify all calls were made
                assert writer.writer.add_scalar.call_count == 6  # 2 training + 3 eval + 1 gradient
                
                writer.close()


class TestMetricLoggerProperties:
    """Property-based tests for MetricLogger class."""
    
    @given(
        loss=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        step=st.integers(min_value=0, max_value=100000)
    )
    def test_log_training_loss_with_various_values(self, loss, step):
        """
        Test that log_training_loss handles various valid loss and step values.
        
        For any valid loss and step values, the logger should handle them
        without raising exceptions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            # This should not raise an exception
            logger.log_training_loss(loss, step)
            
            if writer.is_available():
                writer.close()
    
    @given(
        train_loss=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        val_loss=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        learning_rate=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
        step=st.integers(min_value=0, max_value=100000)
    )
    def test_log_evaluation_metrics_with_various_values(self, train_loss, val_loss, learning_rate, step):
        """
        Test that log_evaluation_metrics handles various valid metric values.
        
        For any valid metric values, the logger should handle them
        without raising exceptions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            # This should not raise an exception
            logger.log_evaluation_metrics(train_loss, val_loss, learning_rate, step)
            
            if writer.is_available():
                writer.close()
    
    @given(
        grad_norm=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        step=st.integers(min_value=0, max_value=100000)
    )
    def test_log_gradient_norm_with_various_values(self, grad_norm, step):
        """
        Test that log_gradient_norm handles various valid gradient norm values.
        
        For any valid gradient norm and step values, the logger should handle them
        without raising exceptions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            # This should not raise an exception
            logger.log_gradient_norm(grad_norm, step)
            
            if writer.is_available():
                writer.close()
    
    def test_error_disables_subsequent_logging(self):
        """
        Test that after a logging error, subsequent logging calls are skipped.
        
        When a logging error occurs and writer.available is set to False,
        subsequent logging calls should be skipped without attempting to log.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = TensorBoardWriter(log_dir)
            logger = MetricLogger(writer)
            
            if writer.is_available():
                # Mock add_scalar to raise an exception on first call
                call_count = [0]
                
                def side_effect(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise RuntimeError("Test error")
                
                writer.writer.add_scalar = MagicMock(side_effect=side_effect)
                
                with patch('builtins.print'):
                    # First call should trigger error and disable writer
                    logger.log_training_loss(0.5, 100)
                    assert not writer.is_available()
                    
                    # Second call should be skipped (no exception raised)
                    logger.log_training_loss(0.4, 101)
                    
                    # Verify add_scalar was only called once
                    assert call_count[0] == 1
                
                writer.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
