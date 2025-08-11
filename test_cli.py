#!/usr/bin/env python3
import sys
import os
sys.path.append('/home/ustad')

# Import the CLI module
from ollama_ISF_Engine_cli import CLIInterface

def test_cli():
    print("Testing CLI functionality...")
    try:
        cli = CLIInterface()
        print("✓ CLI initialized successfully!")
        
        # Test a simple input processing
        print("\nTesting basic text processing...")
        cli.process_input("John is a person. John likes coffee.")
        print("✓ Basic processing works!")
        
        # Test stats
        print("\nTesting stats...")
        cli.show_stats()
        print("✓ Stats work!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cli()
