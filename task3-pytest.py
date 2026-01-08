import pytest

def find_all_with_mismatch(fragment, sequence, protein_id, max_mismatches=1):
    """
    Find all occurrences of fragment in sequence allowing up to max_mismatches.
    
    Args:
        fragment: The k-mer fragment to search for
        sequence: The protein sequence to search in
        protein_id: The ID of the protein
        max_mismatches: Maximum allowed mismatches
    
    Returns:
        List of dicts, each containing:
            - fragment: The k-mer fragment being searched
            - protein_id: The ID of the protein
            - position: Start position of match in sequence
            - matched_sequence: The actual matched sequence
            - mismatches: List of (position, query_char, target_char) tuples
    """
    matches = []
    frag_len = len(fragment)
    
    # Slide through the sequence
    for pos in range(len(sequence) - frag_len + 1):
        candidate = sequence[pos:pos + frag_len]
        
        # Count mismatches between fragment and candidate
        mismatch_list = []
        for i in range(frag_len):
            if fragment[i] != candidate[i]:
                mismatch_list.append((i, fragment[i], candidate[i]))
        
        # Only keep matches within the allowed mismatch threshold
        if len(mismatch_list) <= max_mismatches:
            matches.append({
                "fragment": fragment,
                "protein_id": protein_id,
                "position": pos,
                "matched_sequence": candidate,
                "mismatches": mismatch_list
            })
    
    return matches
    # m = len(B)
    # Time: O(m*k)
    # Space: O(m)


def find_peptide_overlaps(query_sequences, proteome_dict, k=9, max_mismatches=1):
    """
    Find all occurrences of k-mer fragments from query sequences 
    in the proteome database.
    
    Args:
        query_sequences: List of peptide sequences to search for
        proteome_dict: Dict {protein_id: sequence}
        k: Length of fragments to generate
        max_mismatches: Maximum allowed mismatches (default 1)
    
    Returns:
        Dict mapping each query to list of matches
    """
    result = {}
    for A in query_sequences:
        all_matches = []
        for B_id, B in proteome_dict.items():
            for i in range(len(A)-k+1):
                fragment = A[i:i+k]
                matches = find_all_with_mismatch(fragment, B, B_id, max_mismatches=max_mismatches)
                all_matches.extend(matches)
        result[A] = all_matches

    return result
    # h = len(query_sequences)
    # g = len(proteome_dict)
    # Time: O(h * g * n * m * k) = O(10^12)
    # Polynomial time complexity 
    # Space: O(h*g*n*m)


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

