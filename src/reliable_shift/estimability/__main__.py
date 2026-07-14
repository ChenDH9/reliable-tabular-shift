from __future__ import annotations

import argparse
import json

from . import binomial_support_approximation, hypergeometric_support_probability

parser = argparse.ArgumentParser(description="Exact calibration class-support probability")
parser.add_argument("--pool-size", type=int, required=True)
parser.add_argument("--positive-count", type=int, required=True)
parser.add_argument("--budget", type=int, required=True)
parser.add_argument("--min-positive", type=int, required=True)
parser.add_argument("--min-negative", type=int, required=True)
args = parser.parse_args()
payload = vars(args)
payload["exact_hypergeometric"] = hypergeometric_support_probability(
    args.pool_size, args.positive_count, args.budget, args.min_positive, args.min_negative
)
payload["binomial_approximation"] = binomial_support_approximation(
    args.pool_size, args.positive_count, args.budget, args.min_positive, args.min_negative
)
payload["approximation_difference"] = payload["binomial_approximation"] - payload["exact_hypergeometric"]
print(json.dumps(payload, indent=2))
