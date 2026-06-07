import argparse
import sqlite3
import sys
import logging
import os
import json

from commands.upsert import UpsertCommand
from commands.pipeline import PipelineCommand
from languagemodels.embedder import ChunkEmbedder # Updated import
from configs import DB_PATH

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Document Tools CLI")
    parser.add_argument("command", choices=["upsert", "pipeline"], help="Command to run")
    parser.add_argument("-f", "--file", help="Path to file for upsert", nargs="+")
    parser.add_argument("-q", "--query", help="User query")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Initialize Embedder (Dependency for Retrieve)
    embedder = ChunkEmbedder()

    try:
        if args.command == "upsert":
            cmd = UpsertCommand(conn)
            cmd.execute(args.file)
        elif args.command == "pipeline":
            cmd = PipelineCommand(conn)
            print(cmd.execute(input_text=args.query))
    finally:
        conn.close()

if __name__ == "__main__":
    main()
