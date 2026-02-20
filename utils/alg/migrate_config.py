"""
Configuration migration script for ALG1 training project.
Migrates legacy config.json to new split configuration structure.

Usage:
    python migrate_config.py                           # Use defaults
    python migrate_config.py --source my_config.json   # Custom source
    python migrate_config.py --target my_configs       # Custom target directory
"""

import argparse
from utils.alg.utils_config_loader import ConfigMigrator

def main():
    parser = argparse.ArgumentParser(
        description='Migrate legacy config.json to split configuration files'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='config.json',
        help='Path to legacy config.json file (default: config.json)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='configs',
        help='Target directory for split config files (default: configs)'
    )
    args = parser.parse_args()
    
    print(f"Migrating configuration from '{args.source}' to '{args.target}/'...")
    
    migrator = ConfigMigrator()
    migrator.migrate(source=args.source, target_dir=args.target)
    
    print(f"\nMigration complete! Created files:")
    print(f"  - {args.target}/model.json")
    print(f"  - {args.target}/train.yaml")
    print(f"  - {args.target}/data.yaml")
    print(f"\nYou can now use 'python train.py' to train with the new configuration structure.")

if __name__ == '__main__':
    main()
