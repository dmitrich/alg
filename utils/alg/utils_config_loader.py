"""
Configuration management module for ALG1 training project.
Handles loading, merging, and migration of configuration files.
"""

import json
import yaml
import os
from pathlib import Path

class ConfigLoader:
    """Loads and merges configuration from multiple sources."""
    
    def load_configs(self, config_dir: str = "configs") -> dict:
        """
        Load and merge all configuration files.
        
        Args:
            config_dir: Directory containing config files
            
        Returns:
            Merged configuration dictionary
        """
        if os.path.exists(config_dir):
            model_config = self.load_model_config(f"{config_dir}/model.json")
            train_config = self.load_train_config(f"{config_dir}/train.yaml")
            data_config = self.load_data_config(f"{config_dir}/data.yaml")
            
            merged = {}
            merged.update(model_config)
            merged.update(train_config)
            merged.update(data_config)
            
            return merged
        else:
            import json
            with open('config.json', 'r') as f:
                return json.load(f)
    
    def load_model_config(self, path: str = "configs/model.json") -> dict:
        """Load architecture configuration."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_train_config(self, path: str = "configs/train.yaml") -> dict:
        """Load training hyperparameters."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data_config(self, path: str = "configs/data.yaml") -> dict:
        """Load data configuration."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

class ConfigMigrator:
    """Migrates legacy config.json to new structure."""
    
    def migrate(self, source: str = "config.json", target_dir: str = "configs") -> None:
        """
        Migrate config.json to split configuration files.
        
        Args:
            source: Path to legacy config.json
            target_dir: Directory to create new config files
        """
        with open(source, 'r') as f:
            config = json.load(f)
        
        os.makedirs(target_dir, exist_ok=True)
        
        model_params = self.extract_model_params(config)
        with open(f"{target_dir}/model.json", 'w') as f:
            json.dump(model_params, f, indent=2)
        
        train_params = self.extract_train_params(config)
        with open(f"{target_dir}/train.yaml", 'w') as f:
            yaml.dump(train_params, f, default_flow_style=False)
        
        data_config = self.create_default_data_config()
        with open(f"{target_dir}/data.yaml", 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    def extract_model_params(self, config: dict) -> dict:
        """Extract architecture parameters."""
        model_keys = ['n_embd', 'n_head', 'n_layer', 'dropout']
        model_params = {k: config[k] for k in model_keys if k in config}
        
        if 'vocab_size' in config:
            model_params['vocab_size'] = config['vocab_size']
        else:
            model_params['vocab_size'] = 65
        
        return model_params
    
    def extract_train_params(self, config: dict) -> dict:
        """Extract training hyperparameters."""
        train_keys = ['batch_size', 'block_size', 'max_iters', 
                      'eval_interval', 'learning_rate', 'eval_iters']
        return {k: config[k] for k in train_keys if k in config}
    
    def create_default_data_config(self) -> dict:
        """Create default data configuration."""
        return {
            'data_path': 'data/input.txt',
            'train_split': 0.9,
            'val_split': 0.1
        }
