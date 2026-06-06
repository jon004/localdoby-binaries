import argparse
import sqlite3
import sys
import logging
from commands.upsert import UpsertCommand
from commands.retrieve import RetrieveCommand
from commands.prompt import PromptCommand
from languagemodels.generator import LanguageModelClient

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Document Tools CLI")
    parser.add_argument("command", choices=["upsert", "retrieve", "prompt"], help="Command to run")
    parser.add_argument("-f", "--file", help="Path to file for upsert", nargs="+")
    parser.add_argument("-q", "--query", help="Query for retrieve or prompt")
    parser.add_argument("-s", "--system", help="System prompt for prompt command")
    
    args = parser.parse_args()

    # Initialize shared database connection
    db_path = "document_data.db"
    conn = sqlite3.connect(db_path)

    # Initialize the Language Model Client (shared for commands that need it)
    model_client = LanguageModelClient()

    try:
        if args.command == "upsert":
            if not args.file:
                print("Error: 'upsert' requires a file path using -f")
                sys.exit(1)
            cmd = UpsertCommand(conn)
            cmd.execute(args.file)

        elif args.command == "retrieve":
            if not args.query:
                print("Error: 'retrieve' requires a query using -q")
                sys.exit(1)
            # FIXED: Passing model_client here
            cmd = RetrieveCommand(conn, model_client=model_client)
            results = cmd.execute(pivot_query=args.query, attribute_query=args.query)
            for res in results:
                print(res)

        elif args.command == "prompt":
            if not args.query:
                print("Error: 'prompt' requires a prompt using -q")
                sys.exit(1)
            cmd = PromptCommand()
            response = cmd.execute(prompt=args.query, system_prompt=args.system)
            print(response)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
