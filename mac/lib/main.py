import argparse
import sqlite3
import sys
import logging
import os
import json

from commands.upsert import UpsertCommand
from commands.pipeline import PipelineCommand
from commands.prompt import PromptCommand
from languagemodels.embedder import ChunkEmbedder
from configs import DB_PATH

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Document Tools CLI")
    
    # Use subparsers to cleanly separate arguments for different commands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # --- 1. Upsert Command ---
    upsert_parser = subparsers.add_parser("upsert", help="Upsert documents into the database")
    upsert_parser.add_argument("-f", "--file", help="Path to file for upsert", nargs="+", required=True)

    # --- 2. Pipeline Command ---
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the fact verification pipeline")
    pipeline_parser.add_argument("-q", "--query", help="User query", required=True)

    # --- 3. Prompt Command (The Testing Sandbox) ---
    prompt_parser = subparsers.add_parser("prompt", help="Directly prompt a model (One-Shot)")
    
    # Required Core Arguments
    prompt_parser.add_argument("-q", "--query", help="The prompt text to send to the model", required=True)
    prompt_parser.add_argument("-m", "--model", help="Path to the model to use", required=True)
    
    # Optional Routing / Context
    prompt_parser.add_argument("-r", "--role", help="Message role (default: user)", default="user")
    prompt_parser.add_argument("-s", "--system-prompt", help="Inject a system prompt before the query")
    
    # Optional Inference Overrides
    prompt_parser.add_argument("--temp", type=float, help="Temperature (e.g., 0.1)")
    prompt_parser.add_argument("--max-tokens", type=int, help="Max new tokens to generate")
    prompt_parser.add_argument("--penalty", type=float, help="Repetition penalty (e.g., 1.1)")
    prompt_parser.add_argument("--top-p", type=float, help="Top-P sampling (e.g., 0.95)")
    prompt_parser.add_argument("--top-k", type=int, help="Top-K sampling (e.g., 40)")
    prompt_parser.add_argument("--penalty-freq", type=float, help="Frequency penalty")
    prompt_parser.add_argument("--penalty-present", type=float, help="Presence penalty")

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
            result = cmd.execute(input_text=args.query)
            print(json.dumps(result, indent=2))
            
        elif args.command == "prompt":
            cmd = PromptCommand()
            result = cmd.execute(
                prompt=args.query,
                model_path=args.model,
                role=args.role,
                system_prompt=args.system_prompt,
                temperature=args.temp,
                repetition_penalty=args.penalty,
                max_new_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                penalty_freq=args.penalty_freq,
                penalty_present=args.penalty_present
            )
            print("\n=== MODEL OUTPUT ===")
            print(result.get("response", {}).get("message", ""))
            print("====================\n")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
