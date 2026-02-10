"""
Run directory management module for ALG1 training project.
Handles run ID generation, directory creation, and metadata capture.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import shutil
import torch


class RunIDGenerator:
    """Generates unique run identifiers."""
    
    def generate(self, tag: str = None) -> str:
        """
        Generate unique run ID in format YYYY-MM-DD_NNN_tag.
        
        Args:
            tag: Optional short tag to append to run ID
            
        Returns:
            Run ID string (e.g., "2024-01-15_001_baseline")
        """
        date = datetime.now().strftime("%Y-%m-%d")
        counter = self.get_next_counter(date)
        
        run_id = f"{date}_{counter:03d}"
        if tag:
            run_id = f"{run_id}_{tag}"
        
        return run_id
    
    def get_next_counter(self, date: str, runs_dir: str = "runs") -> int:
        """
        Get next counter value for given date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            runs_dir: Directory containing run directories
            
        Returns:
            Next counter value (e.g., 1, 2, 3...)
        """
        if not os.path.exists(runs_dir):
            return 1
        
        # Find all run directories for this date
        existing_runs = []
        for entry in os.listdir(runs_dir):
            if entry.startswith(date):
                # Extract counter from run ID (format: YYYY-MM-DD_NNN or YYYY-MM-DD_NNN_tag)
                parts = entry.split('_')
                if len(parts) >= 2:
                    try:
                        counter = int(parts[1])
                        existing_runs.append(counter)
                    except ValueError:
                        continue
        
        if not existing_runs:
            return 1
        
        return max(existing_runs) + 1


class RunDirectoryCreator:
    """Creates run directory structure."""
    
    def create(self, run_id: str, base_dir: str = "runs") -> str:
        """
        Create complete run directory structure.
        
        Args:
            run_id: Unique run identifier
            base_dir: Base directory for runs
            
        Returns:
            Path to created run directory
        """
        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        self.create_subdirectories(run_dir)
        
        return run_dir
    
    def create_subdirectories(self, run_dir: str) -> None:
        """Create all required subdirectories."""
        subdirs = [
            "meta",
            "logs/tensorboard",
            "artifacts/checkpoints",
            "artifacts/export",
            "eval"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)


class MetadataCapture:
    """Captures git and environment metadata."""
    
    def capture_git_info(self, output_path: str) -> None:
        """
        Capture git commit hash and dirty flag.
        
        Args:
            output_path: Path to write git.txt
        """
        try:
            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                with open(output_path, 'w') as f:
                    f.write("Git information unavailable\n")
                return
            
            commit_hash = result.stdout.strip()
            
            # Check if dirty
            result = subprocess.run(
                ['git', 'diff-index', '--quiet', 'HEAD'],
                capture_output=True,
                timeout=5
            )
            
            is_dirty = result.returncode != 0
            
            # Write git info
            with open(output_path, 'w') as f:
                f.write(f"commit: {commit_hash}\n")
                if is_dirty:
                    f.write("status: dirty\n")
                else:
                    f.write("status: clean\n")
        
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            with open(output_path, 'w') as f:
                f.write("Git command failed\n")
    
    def capture_env_info(self, output_path: str) -> None:
        """
        Capture Python and PyTorch versions.
        
        Args:
            output_path: Path to write env.txt
        """
        python_version = sys.version
        pytorch_version = torch.__version__
        
        with open(output_path, 'w') as f:
            f.write(f"Python: {python_version}\n")
            f.write(f"PyTorch: {pytorch_version}\n")
    
    def copy_configs(self, config_dir: str, meta_dir: str) -> None:
        """
        Copy configuration files to meta directory.
        
        Args:
            config_dir: Source configuration directory
            meta_dir: Target meta directory
        """
        config_files = ['model.json', 'train.yaml', 'data.yaml']
        
        for config_file in config_files:
            src = os.path.join(config_dir, config_file)
            dst = os.path.join(meta_dir, config_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Create empty notes.md file
        notes_path = os.path.join(meta_dir, 'notes.md')
        with open(notes_path, 'w') as f:
            f.write("# Run Notes\n\n")
