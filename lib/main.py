import warnings
warnings.filterwarnings("ignore", category=Warning, module='urllib3.*')

import argparse
import sqlite3
import json
import sys
import os
import logging

from dataclasses import asdict

from commands.upsert import UpsertCommand
from commands.search import SearchCommand
from commands.sliding_prompt import SlidingPromptCommand
from commands.cluster import ClusterCommand
from languagemodels.generator import LanguageModelClient
from configs import (DEFAULT_SIMILARITY_SCORE_FOR_SEARCH_THRESHOLD, DEFAULT_SEARCH_LIMIT,
                    DEFAULT_SLIDING_PROMPT_SIMILARITY_SCORE, DEFAULT_GRANULAR_SIMILARITY_SCORE,
                    DEFAULT_CLUSTER_SIMILARITY_SCORE)

DB_PATH = os.path.expanduser("~/.vectordb/documents.db")
EMBEDDER_MODEL_NAME = "embed-model.gguf"
LANGUAGE_MODEL_URL = "http://0.0.0.0:8080"

def setup_logging(verbose: bool):
    """Setup logging configuration"""    
    level = logging.INFO if verbose else logging.ERROR
    # Set level on root logger so all child loggers inherit it
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(message)s',
        # format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr  # Log to stderr so stdout is clean for JSON output
    )
    return logging.getLogger(__name__)

def main():
    # 0. Parse args early to get verbose flag
    parser = argparse.ArgumentParser(description="Vector Document CLI", add_help=False)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("command", nargs="?", help="Command to execute")
    known_args, _ = parser.parse_known_args()
    
    # Setup logging based on verbose flag
    logger = setup_logging(known_args.verbose)
    
    # 1. Shared Database Connection
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if known_args.verbose:
        logger.info(f"Connected to database: {DB_PATH}")

    # 3. Command Objects
    upsert_cmd = UpsertCommand(conn)
    search_cmd = SearchCommand(conn)
    sliding_prompt_cmd = SlidingPromptCommand(conn)
    cluster_cmd = ClusterCommand()

    parser = argparse.ArgumentParser(description="Vector Document CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command")

    # --- CLI DEFINITIONS ---
    upsert_p = subparsers.add_parser("upsert")
    upsert_p.add_argument("-f", "--files", dest="files", nargs="+")

    search_p = subparsers.add_parser("search")
    search_p.add_argument("-q", "--query", required=True)
    search_p.add_argument("-ff", "--file-filter", nargs="+")
    search_p.add_argument("-l", "--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    search_p.add_argument("-s", "--similarity-score", type=float, default=DEFAULT_SIMILARITY_SCORE_FOR_SEARCH_THRESHOLD, help="Minimum similarity score threshold (default: 0.555)")
    search_p.add_argument("-g", "--single-sentence-granularity", action="store_true", help="Enable single sentence granularity search (default: false)")
    search_p.add_argument("--filter-seen-chunks", action="store_true", help="Skip filtering of duplicate chunks (default: false)")

    sliding_prompt_p = subparsers.add_parser("sliding-prompt")
    sliding_prompt_p.add_argument("-p", "--prompt", required=True)
    sliding_prompt_p.add_argument("-ff", "--file-filter", nargs="+", help="Filter by specific files")
    sliding_prompt_p.add_argument("-m", "--model", help="Model path")
    sliding_prompt_p.add_argument("-t", "--chat-template", help="Chat template")
    sliding_prompt_p.add_argument("-rf", "--rag-filter", action="store_true", help="Enable RAG filtering (default: evaluate all chunks)")
    sliding_prompt_p.add_argument("-s", "--similarity-score", type=float, default=DEFAULT_SLIDING_PROMPT_SIMILARITY_SCORE, help="Minimum similarity score threshold for RAG filtering (default: 0.555)")
    # Granularity Flags
    sliding_prompt_p.add_argument("-g", "--single-sentence-granularity", action="store_true")
    sliding_prompt_p.add_argument("--without-siblings", action="store_true")
    sliding_prompt_p.add_argument("--no-granular-filter", action="store_true")
    sliding_prompt_p.add_argument("--granular-similarity-score", type=float, default=DEFAULT_GRANULAR_SIMILARITY_SCORE, help="Minimum similarity score threshold for granular filtering step (default: 0.5)")

    cluster_p = subparsers.add_parser("cluster")
    cluster_p.add_argument("--chunks", nargs="+", required=True, help="List of sentences")
    cluster_p.add_argument("-s", "--similarity-score", type=float, default=DEFAULT_CLUSTER_SIMILARITY_SCORE, help="Minimum similarity score threshold (default: 0.94)")
    cluster_p.add_argument("-g", "--single-sentence-granularity", action="store_true", help="Enable single sentence granularity clustering (default: false)")

    args = parser.parse_known_args()
    args_dict = vars(args[0])
    
    # Merge early args with full args (verbose flag from early parser takes precedence)
    args_dict['verbose'] = known_args.verbose

    if args_dict['command'] == "upsert":
        upsert_cmd.execute(args_dict['files'])
    elif args_dict['command'] == "search":
        results = search_cmd.execute(args_dict['query'], args_dict['file_filter'], args_dict['limit'], args_dict['similarity_score'], args_dict['single_sentence_granularity'], args_dict['filter_seen_chunks'])
        print(json.dumps([asdict(r) for r in results], indent=2))
    elif args_dict['command'] == "sliding-prompt":
        results = sliding_prompt_cmd.execute(
            query=args_dict['prompt'], 
            file_filters=args_dict['file_filter'] if args_dict['file_filter'] is not None else [], 
            model_path=args_dict['model'], 
            chat_template=args_dict['chat_template'], 
            rag_filter=args_dict['rag_filter'], 
            similarity_score=args_dict['similarity_score'],
            single_sentence_granularity=args_dict['single_sentence_granularity'],
            without_siblings=args_dict['without_siblings'],
            no_granular_filter=args_dict['no_granular_filter'],
            granular_similarity_score=args_dict['granular_similarity_score']
        )
        print(json.dumps(results, indent=2))
    elif args_dict['command'] == "cluster":
        cluster_cmd.threshold = args_dict['similarity_score']
        cluster_cmd.single_sentence_granularity = args_dict['single_sentence_granularity']
        results = cluster_cmd.execute(args_dict['chunks'])
        print(json.dumps(results, indent=2))
    else:
        parser.print_help()

    conn.close()

if __name__ == "__main__":
    main()
