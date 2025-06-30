#!/usr/bin/env python3
"""
Code-Switch Aware AI Library - CLI Entry Point

Usage:
    python cli.py
    
This script provides a command-line interface for testing and using
the code-switch aware AI library features.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codeswitch_ai.interface.cli import main

if __name__ == "__main__":
    main()