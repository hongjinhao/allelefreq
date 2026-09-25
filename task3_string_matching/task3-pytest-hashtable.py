import pytest
from collections import defaultdict
from itertools import combinations, product

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_kmer_index(proteome_dict, k):
    """
    Build a hashtable mapping every k-mer occurring anywhere in the proteome
    to the list of (protein_id, position) where it occurs.

    Built once (per k), not once per query - this is the whole point versus
    brute force, which rescans every protein from scratch for every query.

    Time: O(p * k), where p = total proteome length (sum of all protein lengths)
    Space: O(p)
    """
    index = defaultdict(list)
    for protein_id, sequence in proteome_dict.items():
        for pos in range(len(sequence) - k + 1):
            index[sequence[pos:pos + k]].append((protein_id, pos))
    return index


def generate_mismatch_neighbors(fragment, max_mismatches):
    """
    Yield every string within Hamming distance <= max_mismatches of fragment
    (including fragment itself), built over the 20-letter amino acid alphabet.

    Because each (positions, substitution-letters) combination maps to exactly
    one resulting string, this never yields duplicates.

    Count of variants: sum_{i=0}^{max_mismatches} C(len(fragment), i) * 19^i
    e.g. len=9, max_mismatches=1 -> 1 + 9*19 = 172 - cheap regardless of
    proteome size. Grows combinatorially for larger max_mismatches (still
    fine for 2, impractical much beyond that - use a proper approximate
    matching index such as an FM-index for larger mismatch budgets).
    """
    length = len(fragment)
    max_mismatches = min(max_mismatches, length)
    for num_subs in range(max_mismatches + 1):
        for positions in combinations(range(length), num_subs):
            alt_choices = [[aa for aa in AMINO_ACIDS if aa != fragment[p]] for p in positions]
            for combo in product(*alt_choices):
                variant = list(fragment)
                for p, aa in zip(positions, combo):
                    variant[p] = aa
                yield "".join(variant)


def find_all_with_mismatch(fragment, index, proteome_dict, max_mismatches=1):
    """
    Hashtable equivalent of a brute-force sliding-window scan: instead of
    comparing `fragment` against every position of every protein, generate
    every neighbor within max_mismatches and look each one up directly in
    the prebuilt k-mer index.

    Returns a list of dicts, each containing:
        - fragment: The k-mer fragment being searched
        - protein_id: The ID of the protein
        - position: Start position of match in sequence
        - matched_sequence: The actual matched sequence
        - mismatches: List of (position, query_char, target_char) tuples

    Time: O(V * k) hashtable lookups, where V = number of neighbors - independent
    of proteome size (g proteins, m residues), unlike the brute-force O(m*k) per protein.
    """
    matches = []
    for variant in generate_mismatch_neighbors(fragment, max_mismatches):
        for protein_id, pos in index.get(variant, []):
            candidate = proteome_dict[protein_id][pos:pos + len(fragment)]
            mismatch_list = [
                (i, fragment[i], candidate[i])
                for i in range(len(fragment))
                if fragment[i] != candidate[i]
            ]
            matches.append({
                "fragment": fragment,
                "protein_id": protein_id,
                "position": pos,
                "matched_sequence": candidate,
                "mismatches": mismatch_list,
            })
    return matches


def find_peptide_overlaps(query_sequences, proteome_dict, k=9, max_mismatches=1):
    """
    Find all occurrences of k-mer fragments from query sequences in the
    proteome database, using a prebuilt k-mer hashtable index instead of
    brute-force scanning every protein for every query.

    Args:
        query_sequences: List of peptide sequences to search for
        proteome_dict: Dict {protein_id: sequence}
        k: Length of fragments to generate
        max_mismatches: Maximum allowed mismatches (default 1)

    Returns:
        Dict mapping each query to list of matches
    """
    index = build_kmer_index(proteome_dict, k)

    result = {}
    for A in query_sequences:
        all_matches = []
        for i in range(len(A) - k + 1):
            fragment = A[i:i + k]
            all_matches.extend(
                find_all_with_mismatch(fragment, index, proteome_dict, max_mismatches=max_mismatches)
            )
        result[A] = all_matches

    return result
    # h = len(query_sequences), g = len(proteome_dict), n = len(A)
    # p = total proteome length (sum of all protein lengths), V = neighbor count ~ k*19+1
    # Index build (once per call): O(p * k)
    # Per query: O((n-k+1) * V * k), independent of g and p
    # Total: O(p*k + h*n*k^2) roughly - proteome size no longer multiplies query cost
    # Space: O(p) for the index


@pytest.fixture
def sample_proteome():
    """Sample human proteome with one protein"""
    return {
        "A0A087WZT3": "MELSAEYLREKLQRDLEAEHVLPSPGGVGQVRGETAASETQLGS"
    }

def test_exact_match_single_fragment(sample_proteome):
    """Test finding exact matches for peptide fragments"""
    query_sequences = ["RDLEAEHVLP"]
    k = 9

    results = find_peptide_overlaps(query_sequences, sample_proteome, k=k)

    # Should find matches for both k-mers: RDLEAEHVL and DLEAEHVLP
    assert "RDLEAEHVLP" in results
    assert len(results["RDLEAEHVLP"]) == 2

    # Check first fragment: RDLEAEHVL at position 13
    match1 = results["RDLEAEHVLP"][0]
    assert match1["fragment"] == "RDLEAEHVL"
    assert match1["protein_id"] == "A0A087WZT3"
    assert match1["position"] == 13
    assert match1["matched_sequence"] == "RDLEAEHVL"
    assert match1["mismatches"] == []

    # Check second fragment: DLEAEHVLP at position 14
    match2 = results["RDLEAEHVLP"][1]
    assert match2["fragment"] == "DLEAEHVLP"
    assert match2["protein_id"] == "A0A087WZT3"
    assert match2["position"] == 14
    assert match2["matched_sequence"] == "DLEAEHVLP"
    assert match2["mismatches"] == []

def test_match_with_one_mismatch(sample_proteome):
    """Test finding matches with exactly one mismatch"""
    # MELSAEYLX differs from MELSAEYLR by one character (X vs R at position 8)
    query_sequences = ["MELSAEYLX"]
    k = 9

    results = find_peptide_overlaps(query_sequences, sample_proteome, k=k)

    assert "MELSAEYLX" in results
    assert len(results["MELSAEYLX"]) == 1

    match = results["MELSAEYLX"][0]
    assert match["fragment"] == "MELSAEYLX"
    assert match["protein_id"] == "A0A087WZT3"
    assert match["position"] == 0
    assert match["matched_sequence"] == "MELSAEYLR"
    assert len(match["mismatches"]) == 1
    assert match["mismatches"][0] == (8, 'X', 'R')

def test_no_match_exceeds_mismatch_limit(sample_proteome):
    """Test that sequences with >1 mismatches are not matched"""
    # MELSAEXYZ has 3 mismatches compared to MELSAEYLR
    query_sequences = ["MELSAEXYZ"]
    k = 9

    results = find_peptide_overlaps(query_sequences, sample_proteome, k=k)

    # Should either have no entry or empty list
    if "MELSAEXYZ" in results:
        assert len(results["MELSAEXYZ"]) == 0

def test_multiple_queries(sample_proteome):
    """Test processing multiple query sequences"""
    query_sequences = ["RDLEAEHVLP", "MELSAEYLX"]
    k = 9

    results = find_peptide_overlaps(query_sequences, sample_proteome, k=k)

    # Both queries should have results
    assert "RDLEAEHVLP" in results
    assert "MELSAEYLX" in results
    assert len(results["RDLEAEHVLP"]) == 2  # Two fragments match
    assert len(results["MELSAEYLX"]) == 1   # One fragment with mismatch

def test_different_k_values(sample_proteome):
    """Test with different k-mer lengths"""
    query_sequences = ["RDLEAEHVLP"]

    # Test with k=8
    results_k8 = find_peptide_overlaps(query_sequences, sample_proteome, k=8)
    # With k=8, we get 3 fragments: RDLEAEHV, DLEAEHVL, LEAEHVLP
    assert len(results_k8["RDLEAEHVLP"]) == 3

    # Test with k=10
    results_k10 = find_peptide_overlaps(query_sequences, sample_proteome, k=10)
    # With k=10, we get 1 fragment: RDLEAEHVLP (entire sequence)
    assert len(results_k10["RDLEAEHVLP"]) == 1

def test_no_matches_found(sample_proteome):
    """Test when query sequence has no matches in proteome"""
    query_sequences = ["ZZZZZZZZZ"]  # Unlikely to match
    k = 9

    results = find_peptide_overlaps(query_sequences, sample_proteome, k=k)

    if "ZZZZZZZZZ" in results:
        assert len(results["ZZZZZZZZZ"]) == 0

def test_query_shorter_than_k(sample_proteome):
    """Test handling of query sequences shorter than k"""
    query_sequences = ["ABCD"]  # Only 4 amino acids
    k = 9

    results = find_peptide_overlaps(query_sequences, sample_proteome, k=k)

    # Should handle gracefully - either no entry or empty results
    if "ABCD" in results:
        assert len(results["ABCD"]) == 0

def test_edge_case_exact_k_length_query(sample_proteome):
    """Test when query length exactly equals k"""
    query_sequences = ["MELSAEYLR"]  # Exactly 9 amino acids
    k = 9

    results = find_peptide_overlaps(query_sequences, sample_proteome, k=k)

    # Should generate exactly 1 fragment and find 1 exact match
    assert "MELSAEYLR" in results
    assert len(results["MELSAEYLR"]) == 1
    assert results["MELSAEYLR"][0]["position"] == 0
    assert results["MELSAEYLR"][0]["mismatches"] == []
