"""Loads query_peptides.csv into the query_sequences list shape expected by
find_peptide_overlaps() in task3-pytest-hashtable.py / task3-pytest-bruteforce.py.
"""
import csv
import os

_CSV_PATH = os.path.join(os.path.dirname(__file__), "query_peptides.csv")


def load_query_peptides(csv_path=_CSV_PATH):
    """Returns (query_sequences, query_metadata) where query_sequences is a
    plain list of amino-acid strings and query_metadata maps sequence -> row dict.
    """
    query_sequences = []
    query_metadata = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seq = row["sequence"]
            query_sequences.append(seq)
            query_metadata[seq] = row
    return query_sequences, query_metadata


query_sequences, query_metadata = load_query_peptides()

if __name__ == "__main__":
    for seq in query_sequences:
        meta = query_metadata[seq]
        print(f"{meta['name']:45s} ({len(seq):2d} aa)  {seq}")
