"""
Benchmarks the hashtable-indexed k-mer matcher (task3-pytest-hashtable.py)
against the brute-force matcher (task3-pytest-bruteforce.py) on the real
proteome (uniprotkb_human_ref_proteome_dict.pkl, 83,413 proteins).

This is a standalone script, not a pytest suite - pytest only exercises the
tiny single-protein `sample_proteome` fixture in the two task3-pytest-*.py
files, which is why running `pytest` finishes in milliseconds. Real-proteome
timing has to be measured separately, which is what this script does.

Usage:
    python3 task3_benchmark.py
    python3 task3_benchmark.py --k 9 --max-mismatches 1
    python3 task3_benchmark.py --include-brute-force --brute-force-limit 2000
    python3 task3_benchmark.py --queries-file query_peptides.csv
"""
import argparse
import csv
import importlib.util
import itertools
import pickle
import time

DEFAULT_PROTEOME_PATH = "uniprotkb_human_ref_proteome_dict.pkl"

# Same queries already validated by the pytest suite - kept as the default
# so this benchmark is measuring the exact code paths the tests cover,
# just against the real proteome instead of the tiny test fixture.
DEFAULT_QUERIES = ["RDLEAEHVLP", "MELSAEYLX", "MELSAEYLR"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_queries_from_csv(path):
    with open(path, newline="") as f:
        return [row["sequence"] for row in csv.DictReader(f)]


def benchmark_hashtable(module, proteome, queries, k, max_mismatches):
    t0 = time.perf_counter()
    index = module.build_kmer_index(proteome, k)
    build_time = time.perf_counter() - t0

    per_query = {}
    for query in queries:
        t0 = time.perf_counter()
        matches = []
        for i in range(len(query) - k + 1):
            fragment = query[i:i + k]
            matches.extend(
                module.find_all_with_mismatch(fragment, index, proteome, max_mismatches=max_mismatches)
            )
        per_query[query] = {"time": time.perf_counter() - t0, "hits": len(matches)}

    return {"index_build_time": build_time, "index_size": len(index), "per_query": per_query}


def benchmark_brute_force(module, proteome, queries, k, max_mismatches, protein_limit):
    limited_proteome = dict(itertools.islice(proteome.items(), protein_limit))

    per_query = {}
    for query in queries:
        t0 = time.perf_counter()
        result = module.find_peptide_overlaps([query], limited_proteome, k=k, max_mismatches=max_mismatches)
        per_query[query] = {"time": time.perf_counter() - t0, "hits": len(result[query])}

    return {"protein_count": len(limited_proteome), "per_query": per_query}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proteome", default=DEFAULT_PROTEOME_PATH, help="Path to proteome pickle")
    parser.add_argument("--k", type=int, default=9, help="k-mer fragment length")
    parser.add_argument("--max-mismatches", type=int, default=1, help="Max allowed mismatches")
    parser.add_argument("--queries-file", help="CSV with a 'sequence' column (e.g. query_peptides.csv); overrides defaults")
    parser.add_argument(
        "--include-brute-force", action="store_true",
        help="Also time the brute-force matcher. SLOW - capped to --brute-force-limit proteins, "
             "not the full proteome, or it would take minutes per query.",
    )
    parser.add_argument(
        "--brute-force-limit", type=int, default=2000,
        help="Number of proteins to brute-force against (only with --include-brute-force)",
    )
    args = parser.parse_args()

    queries = load_queries_from_csv(args.queries_file) if args.queries_file else DEFAULT_QUERIES

    print(f"Loading proteome from {args.proteome} ...")
    t0 = time.perf_counter()
    with open(args.proteome, "rb") as f:
        proteome = pickle.load(f)
    print(f"  {len(proteome)} proteins loaded in {time.perf_counter() - t0:.2f}s")
    print(f"k={args.k}  max_mismatches={args.max_mismatches}  queries={len(queries)}\n")

    hashtable_module = load_module("task3-pytest-hashtable.py", "task3_hashtable")
    ht_results = benchmark_hashtable(hashtable_module, proteome, queries, args.k, args.max_mismatches)

    print("=== Hashtable-indexed matcher (full proteome) ===")
    print(f"Index build (one-time): {ht_results['index_build_time']:.2f}s  "
          f"({ht_results['index_size']:,} distinct k-mers indexed)")
    for query, stats in ht_results["per_query"].items():
        print(f"  {query!r:30s} {stats['hits']:4d} hits   {stats['time']*1000:8.3f} ms")

    if args.include_brute_force:
        print(f"\n=== Brute-force matcher (capped to {args.brute_force_limit} proteins - "
              f"full proteome would be impractically slow) ===")
        bruteforce_module = load_module("task3-pytest-bruteforce.py", "task3_bruteforce")
        bf_results = benchmark_brute_force(
            bruteforce_module, proteome, queries, args.k, args.max_mismatches, args.brute_force_limit
        )
        for query, stats in bf_results["per_query"].items():
            print(f"  {query!r:30s} {stats['hits']:4d} hits   {stats['time']*1000:8.3f} ms")
        print(f"\n(Brute force ran against {bf_results['protein_count']} of {len(proteome)} proteins; "
              f"extrapolate roughly linearly to estimate full-proteome cost.)")
    else:
        print("\n(Skipped brute-force comparison - pass --include-brute-force to run it, "
              "capped at --brute-force-limit proteins since the full proteome is too slow.)")


if __name__ == "__main__":
    main()
