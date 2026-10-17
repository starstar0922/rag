"""命令行评测：python -m scripts.evaluate --collection hr --size 20"""

import argparse
import json

from app.rag.evaluate import run_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--collection", default="default")
parser.add_argument("--size", type=int, default=20)
parser.add_argument("--no-ragas", action="store_true")
args = parser.parse_args()

result = run_evaluation(args.collection, args.size, use_ragas=not args.no_ragas)
print(json.dumps(result.get("metrics", result), ensure_ascii=False, indent=2))
