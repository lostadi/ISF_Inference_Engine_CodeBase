#!/usr/bin/env python3

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Simple CLI Test")
    parser.add_argument('-q', '--query', type=str, help='Query to process')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.query:
        print(f"Processing query: {args.query}")
        print("Response: This is a test response from the CLI version!")
        print("Logical formulas: Person(Socrates), Mortal(x) ← Man(x)")
        print("Knowledge base updated successfully.")
    elif args.interactive:
        print("🧠 Welcome to the Simple CLI Test!")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("Available commands: help, quit, or enter any text")
                elif user_input:
                    print(f"Processing: {user_input}")
                    print("Response: This is a test response!")
                else:
                    print("Please enter some text or type 'help'")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
