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
from commands.prompt import PromptCommand
from commands.pipeline import PipelineCommand
from commands.retrieve import RetrieveCommand
from configs import (DEFAULT_SIMILARITY_SCORE_FOR_SEARCH_THRESHOLD, DEFAULT_SEARCH_LIMIT,
                    DEFAULT_SLIDING_PROMPT_SIMILARITY_SCORE, DEFAULT_GRANULAR_SIMILARITY_SCORE,
                    DEFAULT_CLUSTER_SIMILARITY_SCORE, DEFAULT_RERANK_THRESHOLD)

# UPDATED: Database path relocated to the standard localdoby directory[cite: 19]
DB_PATH = os.path.expanduser("~/.localdoby/db/localdoby.db")

def setup_logging(verbose: bool):
    """Setup logging configuration"""    
    level = logging.INFO if verbose else logging.ERROR
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(message)s',
        stream=sys.stderr
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
    
    # 1. Shared Database Connection[cite: 2]
    # Ensure the directory exists as defined in dev_install.sh[cite: 19]
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if known_args.verbose:
        logger.info(f"Connected to database: {DB_PATH}")

    # 3. Command Objects
    upsert_cmd = UpsertCommand(conn)
    search_cmd = SearchCommand(conn)
    sliding_prompt_cmd = SlidingPromptCommand(conn)
    cluster_cmd = ClusterCommand()
    prompt_cmd = PromptCommand()
    retrieve_cmd = RetrieveCommand(conn)
    pipeline_cmd = PipelineCommand(conn)

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
    search_p.add_argument("-s", "--similarity-score", type=float, default=DEFAULT_SIMILARITY_SCORE_FOR_SEARCH_THRESHOLD)
    search_p.add_argument("-g", "--single-sentence-granularity", action="store_true")
    search_p.add_argument("--filter-seen-chunks", action="store_true")

    sliding_prompt_p = subparsers.add_parser("sliding-prompt")
    sliding_prompt_p.add_argument("-p", "--prompt", required=True)
    sliding_prompt_p.add_argument("--system-prompt", help="System prompt to prefill before prompting")
    sliding_prompt_p.add_argument("-ff", "--file-filter", nargs="+")
    sliding_prompt_p.add_argument("-m", "--model", help="Model path")
    sliding_prompt_p.add_argument("-t", "--chat-template", help="Chat template")
    sliding_prompt_p.add_argument("-rf", "--rag-filter", action="store_true")
    sliding_prompt_p.add_argument("-s", "--similarity-score", type=float, default=DEFAULT_SLIDING_PROMPT_SIMILARITY_SCORE)
    sliding_prompt_p.add_argument("-g", "--single-sentence-granularity", action="store_true")
    sliding_prompt_p.add_argument("--without-siblings", action="store_true")
    sliding_prompt_p.add_argument("--no-granular-filter", action="store_true")
    sliding_prompt_p.add_argument("--granular-similarity-score", type=float, default=DEFAULT_GRANULAR_SIMILARITY_SCORE)

    cluster_p = subparsers.add_parser("cluster")
    cluster_p.add_argument("--chunks", nargs="+", required=True)
    cluster_p.add_argument("-s", "--similarity-score", type=float, default=DEFAULT_CLUSTER_SIMILARITY_SCORE)
    cluster_p.add_argument("-g", "--single-sentence-granularity", action="store_true")

    prompt_p = subparsers.add_parser("prompt")
    prompt_p.add_argument("-p", "--prompt", required=True)
    prompt_p.add_argument("--system-prompt", help="System prompt to prefill before prompting")
    prompt_p.add_argument("-m", "--model", help="Model path")
    prompt_p.add_argument("-t", "--chat-template", help="Chat template")
    prompt_p.add_argument("--do-not-reset-context", action="store_true")

    # Updated Phase 3: Hybrid Retrieval Parser[cite: 1, 4]
    retrieve_p = subparsers.add_parser("retrieve")
    retrieve_p.add_argument("-pq", "--pivot-query", required=True, help="Keyword-focused query for BM25")
    retrieve_p.add_argument("-aq", "--attribute-query", required=True, help="Semantic query for Vector Search")
    retrieve_p.add_argument("-l", "--limit", type=int, default=30)

    # Updated Phase 1-7: Pipeline Parser[cite: 1]
    pipeline_p = subparsers.add_parser("pipeline")
    pipeline_p.add_argument("--input", required=True, help="Raw text input for extraction")
    pipeline_p.add_argument("--ff", "--file-filter", nargs="+", help="RAG sources")
    pipeline_p.add_argument("--fact-model", default="fact-extractor-1.7b")
    pipeline_p.add_argument("--query-model", default="query-generator-1.5b")
    pipeline_p.add_argument("--judge-model", default="fact-judge-1.7b")
    pipeline_p.add_argument("--rerank-threshold", type=float, default=DEFAULT_RERANK_THRESHOLD)

    args = parser.parse_known_args()
    args_dict = vars(args[0])
    args_dict['verbose'] = known_args.verbose

    if args_dict['command'] == "upsert":
        upsert_cmd.execute(args_dict['files'])
    elif args_dict['command'] == "search":
        results = search_cmd.execute(
            args_dict['query'], 
            args_dict['file_filter'], 
            args_dict['limit'], 
            args_dict['similarity_score'], 
            args_dict['single_sentence_granularity'], 
            args_dict['filter_seen_chunks']
        )
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
            granular_similarity_score=args_dict['granular_similarity_score'],
            system_prompt=args_dict.get('system_prompt')
        )
        print(json.dumps(results, indent=2))
    elif args_dict['command'] == "cluster":
        cluster_cmd.threshold = args_dict['similarity_score']
        cluster_cmd.single_sentence_granularity = args_dict['single_sentence_granularity']
        results = cluster_cmd.execute(args_dict['chunks'])
        print(json.dumps(results, indent=2))
    elif args_dict['command'] == "prompt":
        results = prompt_cmd.execute(
            prompt=args_dict['prompt'],
            system_prompt=args_dict.get('system_prompt'),
            model_path=args_dict.get('model'),
            chat_template=args_dict.get('chat_template'),
            do_not_reset_context=args_dict.get('do_not_reset_context', False)
        )
        print(json.dumps(results, indent=2))
    elif args_dict['command'] == "retrieve":
        # Hybrid retrieval using distinct queries[cite: 1, 4]
        results = retrieve_cmd.execute(
            pivot_query=args_dict['pivot_query'],
            attribute_query=args_dict['attribute_query'],
            top_k=args_dict['limit']
        )
        print(json.dumps(results, indent=2))
    elif args_dict['command'] == "pipeline":
        # Full 7-Phase Execution[cite: 1, 14]
        results = pipeline_cmd.execute(
            input_text=args_dict['input'],
            file_filters=args_dict['ff'] or [],
            fact_model=args_dict['fact_model'],
            query_model=args_dict['query_model'],
            judge_model=args_dict['judge_model'],
            rerank_threshold=args_dict['rerank_threshold']
        )
        print(json.dumps(results, indent=2))
    else:
        parser.print_help()

    conn.close()

if __name__ == "__main__":
    main()
