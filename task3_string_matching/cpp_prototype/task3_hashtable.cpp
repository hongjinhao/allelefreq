// C++ port of task3-pytest-hashtable.py's hashtable-indexed k-mer matcher,
// using a flat CSR-style index (sorted key array + offset array + packed
// position array) instead of a dict-of-lists, to avoid Python's per-object
// boxing overhead. See TASK3_STRING_MATCHING.md for background.
//
// Build:  g++ -O2 -std=c++17 -o task3_hashtable task3_hashtable.cpp
// Run:    ./task3_hashtable [path/to/proteome.bin]

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

// ---------------------------------------------------------------------
// Amino acid encoding: pack each k-mer into a uint64_t, 5 bits/residue.
// PROTEOME_ALPHABET covers every character actually observed in
// uniprotkb_human_ref_proteome_dict.pkl (20 standard residues + U, X).
// QUERY_ALPHABET is the 20-letter substitution set used when generating
// mismatch neighbors -- matches AMINO_ACIDS in task3-pytest-hashtable.py.
// ---------------------------------------------------------------------
static const string PROTEOME_ALPHABET = "ACDEFGHIKLMNPQRSTVWYUX";
static const string QUERY_ALPHABET = "ACDEFGHIKLMNPQRSTVWY";
static int8_t ENCODE[256];
static char DECODE[32];

void init_alphabet() {
    memset(ENCODE, -1, sizeof(ENCODE));
    for (size_t i = 0; i < PROTEOME_ALPHABET.size(); ++i) {
        ENCODE[(unsigned char)PROTEOME_ALPHABET[i]] = (int8_t)i;
        DECODE[i] = PROTEOME_ALPHABET[i];
    }
}

// Returns false if any character in s[0..k) is outside PROTEOME_ALPHABET.
bool encode_kmer(const char* s, int k, uint64_t& out) {
    uint64_t code = 0;
    for (int i = 0; i < k; ++i) {
        int8_t v = ENCODE[(unsigned char)s[i]];
        if (v < 0) return false;
        code = (code << 5) | (uint64_t)v;
    }
    out = code;
    return true;
}

// ---------------------------------------------------------------------
// Proteome: loaded from the flat binary file written by export_proteome.py
// ---------------------------------------------------------------------
struct Proteome {
    vector<string> ids;
    vector<string> seqs;
};

Proteome load_proteome_binary(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) { cerr << "cannot open " << path << "\n"; exit(1); }
    uint32_t n;
    f.read((char*)&n, 4);
    Proteome p;
    p.ids.reserve(n);
    p.seqs.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t id_len;
        f.read((char*)&id_len, 1);
        string id(id_len, '\0');
        f.read(&id[0], id_len);
        uint32_t seq_len;
        f.read((char*)&seq_len, 4);
        string seq(seq_len, '\0');
        f.read(&seq[0], seq_len);
        p.ids.push_back(move(id));
        p.seqs.push_back(move(seq));
    }
    return p;
}

Proteome make_sample_proteome() {
    Proteome p;
    p.ids = {"A0A087WZT3"};
    p.seqs = {"MELSAEYLREKLQRDLEAEHVLPSPGGVGQVRGETAASETQLGS"};
    return p;
}

// ---------------------------------------------------------------------
// Flat CSR-style k-mer index:
//   unique_keys[row]      -- sorted, one entry per distinct k-mer
//   offsets[row..row+1)   -- range into `positions` for that k-mer
//   positions[]           -- one contiguous array of (protein_idx, pos)
// plus a flat open-addressing table mapping key -> row, for O(1) lookup.
// ---------------------------------------------------------------------
struct PositionEntry {
    uint32_t protein_idx;
    uint32_t pos;
};

struct KmerIndex {
    int k = 0;
    vector<uint64_t> unique_keys;
    vector<uint32_t> offsets;
    vector<PositionEntry> positions;

    vector<uint64_t> table_keys;   // EMPTY_KEY = unused slot
    vector<uint32_t> table_rows;
    size_t table_mask = 0;

    static constexpr uint64_t EMPTY_KEY = ~0ULL;  // k*5 <= 60 bits for k<=12, never collides with a real code
};

static inline uint64_t hash_u64(uint64_t x) {
    // splitmix64 finalizer -- fast, good avalanche for integer keys
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

KmerIndex build_kmer_index(const Proteome& proteome, int k) {
    assert(k * 5 <= 60 && "k too large for 5-bit-per-residue uint64 packing");
    KmerIndex idx;
    idx.k = k;

    // Pass 1: collect every (kmer, protein_idx, pos) triple.
    size_t total_residues = 0;
    for (auto& s : proteome.seqs) total_residues += s.size();

    vector<pair<uint64_t, PositionEntry>> entries;
    entries.reserve(total_residues);

    for (uint32_t pi = 0; pi < proteome.seqs.size(); ++pi) {
        const string& seq = proteome.seqs[pi];
        if ((int)seq.size() < k) continue;
        for (uint32_t pos = 0; pos + k <= seq.size(); ++pos) {
            uint64_t code;
            if (encode_kmer(seq.data() + pos, k, code))
                entries.push_back({code, PositionEntry{pi, pos}});
        }
    }

    // Pass 2: sort by kmer code so identical k-mers become contiguous.
    sort(entries.begin(), entries.end(),
         [](const auto& a, const auto& b) { return a.first < b.first; });

    // Pass 3: collapse into CSR layout.
    idx.positions.reserve(entries.size());
    for (size_t i = 0; i < entries.size();) {
        uint64_t key = entries[i].first;
        idx.unique_keys.push_back(key);
        idx.offsets.push_back((uint32_t)idx.positions.size());
        size_t j = i;
        while (j < entries.size() && entries[j].first == key) {
            idx.positions.push_back(entries[j].second);
            ++j;
        }
        i = j;
    }
    idx.offsets.push_back((uint32_t)idx.positions.size());  // end sentinel

    // Pass 4: build the flat open-addressing table (key -> row), load factor ~0.7.
    size_t n = idx.unique_keys.size();
    size_t target = (size_t)(n / 0.7) + 1;
    size_t table_size = 1;
    while (table_size < target) table_size <<= 1;

    idx.table_keys.assign(table_size, KmerIndex::EMPTY_KEY);
    idx.table_rows.assign(table_size, 0);
    idx.table_mask = table_size - 1;

    for (uint32_t row = 0; row < n; ++row) {
        uint64_t key = idx.unique_keys[row];
        size_t slot = hash_u64(key) & idx.table_mask;
        while (idx.table_keys[slot] != KmerIndex::EMPTY_KEY)
            slot = (slot + 1) & idx.table_mask;
        idx.table_keys[slot] = key;
        idx.table_rows[slot] = row;
    }

    return idx;
}

bool lookup(const KmerIndex& idx, uint64_t key, uint32_t& start, uint32_t& end) {
    size_t slot = hash_u64(key) & idx.table_mask;
    while (idx.table_keys[slot] != KmerIndex::EMPTY_KEY) {
        if (idx.table_keys[slot] == key) {
            uint32_t row = idx.table_rows[slot];
            start = idx.offsets[row];
            end = idx.offsets[row + 1];
            return true;
        }
        slot = (slot + 1) & idx.table_mask;
    }
    return false;
}

// ---------------------------------------------------------------------
// Mismatch-neighbor generation: every string within Hamming distance
// <= max_mismatches of `fragment`, built the same way as Python's
// generate_mismatch_neighbors (combinations of positions x product of
// substitution letters), just emitting encoded uint64 keys directly.
// ---------------------------------------------------------------------
static void gen_variants_recursive(const string& fragment, int r, int start_pos,
                                    vector<int>& chosen,
                                    const function<void(uint64_t)>& cb) {
    int k = (int)fragment.size();
    if ((int)chosen.size() == r) {
        int m = r;
        if (m == 0) {
            uint64_t code;
            if (encode_kmer(fragment.data(), k, code)) cb(code);
            return;
        }
        vector<string> choices(m);
        for (int i = 0; i < m; ++i) {
            char orig = fragment[chosen[i]];
            string opts;
            for (char c : QUERY_ALPHABET) if (c != orig) opts += c;
            choices[i] = opts;
        }
        vector<int> idxs(m, 0);
        string variant = fragment;
        while (true) {
            for (int i = 0; i < m; ++i) variant[chosen[i]] = choices[i][idxs[i]];
            uint64_t code;
            if (encode_kmer(variant.data(), k, code)) cb(code);
            int i = m - 1;
            while (i >= 0) {
                if (++idxs[i] < (int)choices[i].size()) break;
                idxs[i] = 0;
                --i;
            }
            if (i < 0) break;
        }
        return;
    }
    for (int p = start_pos; p < k; ++p) {
        chosen.push_back(p);
        gen_variants_recursive(fragment, r, p + 1, chosen, cb);
        chosen.pop_back();
    }
}

void for_each_mismatch_neighbor(const string& fragment, int max_mismatches,
                                 const function<void(uint64_t)>& cb) {
    int k = (int)fragment.size();
    max_mismatches = min(max_mismatches, k);
    for (int r = 0; r <= max_mismatches; ++r) {
        vector<int> chosen;
        gen_variants_recursive(fragment, r, 0, chosen, cb);
    }
}

// ---------------------------------------------------------------------
// Query-facing API, mirroring find_all_with_mismatch / find_peptide_overlaps
// in task3-pytest-hashtable.py.
// ---------------------------------------------------------------------
struct MatchResult {
    string fragment;
    string protein_id;
    uint32_t position;
    string matched_sequence;
    vector<tuple<int, char, char>> mismatches;
};

vector<MatchResult> find_all_with_mismatch(const string& fragment, const KmerIndex& idx,
                                            const Proteome& proteome, int max_mismatches) {
    vector<MatchResult> results;
    int k = (int)fragment.size();
    for_each_mismatch_neighbor(fragment, max_mismatches, [&](uint64_t code) {
        uint32_t start, end;
        if (!lookup(idx, code, start, end)) return;
        for (uint32_t i = start; i < end; ++i) {
            const PositionEntry& e = idx.positions[i];
            const string& seq = proteome.seqs[e.protein_idx];
            string candidate = seq.substr(e.pos, k);
            vector<tuple<int, char, char>> mismatches;
            for (int j = 0; j < k; ++j)
                if (fragment[j] != candidate[j]) mismatches.push_back({j, fragment[j], candidate[j]});
            results.push_back({fragment, proteome.ids[e.protein_idx], e.pos, candidate, move(mismatches)});
        }
    });
    return results;
}

unordered_map<string, vector<MatchResult>> find_peptide_overlaps(
    const vector<string>& query_sequences, const KmerIndex& idx, const Proteome& proteome,
    int k, int max_mismatches) {
    unordered_map<string, vector<MatchResult>> result;
    for (const auto& q : query_sequences) {
        vector<MatchResult> all_matches;
        for (int i = 0; i + k <= (int)q.size(); ++i) {
            string fragment = q.substr(i, k);
            auto matches = find_all_with_mismatch(fragment, idx, proteome, max_mismatches);
            for (auto& m : matches) all_matches.push_back(move(m));
        }
        result[q] = move(all_matches);
    }
    return result;
}

// ---------------------------------------------------------------------
// Correctness tests -- direct C++ port of the 8 pytest cases in
// task3-pytest-hashtable.py, run against the same 1-protein sample fixture.
// ---------------------------------------------------------------------
static int g_failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                      \
            cerr << "  FAIL: " << msg << " (" << #cond << ")\n";            \
            ++g_failures;                                                    \
        }                                                                     \
    } while (0)

void test_exact_match_single_fragment(const Proteome& p) {
    cout << "test_exact_match_single_fragment... ";
    auto idx = build_kmer_index(p, 9);
    auto results = find_peptide_overlaps({"RDLEAEHVLP"}, idx, p, 9, 1);
    auto& m = results["RDLEAEHVLP"];
    CHECK(m.size() == 2, "expected 2 matches");
    if (m.size() == 2) {
        CHECK(m[0].fragment == "RDLEAEHVL", "fragment0");
        CHECK(m[0].protein_id == "A0A087WZT3", "protein_id0");
        CHECK(m[0].position == 13, "position0");
        CHECK(m[0].matched_sequence == "RDLEAEHVL", "matched_sequence0");
        CHECK(m[0].mismatches.empty(), "mismatches0 empty");
        CHECK(m[1].fragment == "DLEAEHVLP", "fragment1");
        CHECK(m[1].position == 14, "position1");
        CHECK(m[1].matched_sequence == "DLEAEHVLP", "matched_sequence1");
        CHECK(m[1].mismatches.empty(), "mismatches1 empty");
    }
    cout << (g_failures == 0 ? "ok\n" : "\n");
}

void test_match_with_one_mismatch(const Proteome& p) {
    cout << "test_match_with_one_mismatch... ";
    int before = g_failures;
    auto idx = build_kmer_index(p, 9);
    auto results = find_peptide_overlaps({"MELSAEYLX"}, idx, p, 9, 1);
    auto& m = results["MELSAEYLX"];
    CHECK(m.size() == 1, "expected 1 match");
    if (m.size() == 1) {
        CHECK(m[0].position == 0, "position");
        CHECK(m[0].matched_sequence == "MELSAEYLR", "matched_sequence");
        CHECK(m[0].mismatches.size() == 1, "1 mismatch");
        if (m[0].mismatches.size() == 1)
            CHECK(m[0].mismatches[0] == make_tuple(8, 'X', 'R'), "mismatch tuple");
    }
    cout << (g_failures == before ? "ok\n" : "\n");
}

void test_no_match_exceeds_mismatch_limit(const Proteome& p) {
    cout << "test_no_match_exceeds_mismatch_limit... ";
    int before = g_failures;
    auto idx = build_kmer_index(p, 9);
    auto results = find_peptide_overlaps({"MELSAEXYZ"}, idx, p, 9, 1);
    CHECK(results["MELSAEXYZ"].empty(), "expected no matches");
    cout << (g_failures == before ? "ok\n" : "\n");
}

void test_multiple_queries(const Proteome& p) {
    cout << "test_multiple_queries... ";
    int before = g_failures;
    auto idx = build_kmer_index(p, 9);
    auto results = find_peptide_overlaps({"RDLEAEHVLP", "MELSAEYLX"}, idx, p, 9, 1);
    CHECK(results["RDLEAEHVLP"].size() == 2, "RDLEAEHVLP has 2");
    CHECK(results["MELSAEYLX"].size() == 1, "MELSAEYLX has 1");
    cout << (g_failures == before ? "ok\n" : "\n");
}

void test_different_k_values(const Proteome& p) {
    cout << "test_different_k_values... ";
    int before = g_failures;
    auto idx8 = build_kmer_index(p, 8);
    auto r8 = find_peptide_overlaps({"RDLEAEHVLP"}, idx8, p, 8, 1);
    CHECK(r8["RDLEAEHVLP"].size() == 3, "k=8 -> 3 fragments");

    auto idx10 = build_kmer_index(p, 10);
    auto r10 = find_peptide_overlaps({"RDLEAEHVLP"}, idx10, p, 10, 1);
    CHECK(r10["RDLEAEHVLP"].size() == 1, "k=10 -> 1 fragment");
    cout << (g_failures == before ? "ok\n" : "\n");
}

void test_no_matches_found(const Proteome& p) {
    cout << "test_no_matches_found... ";
    int before = g_failures;
    auto idx = build_kmer_index(p, 9);
    auto results = find_peptide_overlaps({"ZZZZZZZZZ"}, idx, p, 9, 1);
    CHECK(results["ZZZZZZZZZ"].empty(), "expected no matches");
    cout << (g_failures == before ? "ok\n" : "\n");
}

void test_query_shorter_than_k(const Proteome& p) {
    cout << "test_query_shorter_than_k... ";
    int before = g_failures;
    auto idx = build_kmer_index(p, 9);
    auto results = find_peptide_overlaps({"ABCD"}, idx, p, 9, 1);
    CHECK(results["ABCD"].empty(), "expected no matches");
    cout << (g_failures == before ? "ok\n" : "\n");
}

void test_edge_case_exact_k_length_query(const Proteome& p) {
    cout << "test_edge_case_exact_k_length_query... ";
    int before = g_failures;
    auto idx = build_kmer_index(p, 9);
    auto results = find_peptide_overlaps({"MELSAEYLR"}, idx, p, 9, 1);
    auto& m = results["MELSAEYLR"];
    CHECK(m.size() == 1, "expected 1 match");
    if (m.size() == 1) {
        CHECK(m[0].position == 0, "position 0");
        CHECK(m[0].mismatches.empty(), "no mismatches");
    }
    cout << (g_failures == before ? "ok\n" : "\n");
}

void run_all_tests() {
    Proteome sample = make_sample_proteome();
    cout << "=== correctness tests (port of the 8 pytest cases) ===\n";
    test_exact_match_single_fragment(sample);
    test_match_with_one_mismatch(sample);
    test_no_match_exceeds_mismatch_limit(sample);
    test_multiple_queries(sample);
    test_different_k_values(sample);
    test_no_matches_found(sample);
    test_query_shorter_than_k(sample);
    test_edge_case_exact_k_length_query(sample);
    cout << (g_failures == 0 ? "\nALL TESTS PASSED\n\n" : "\nSOME TESTS FAILED\n\n");
}

// ---------------------------------------------------------------------
// Memory measurement helper (peak RSS), analogous to Python's
// resource.getrusage(...).ru_maxrss used in the earlier benchmark.
// ---------------------------------------------------------------------
size_t get_vm_hwm_kb() {
    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            size_t kb;
            sscanf(line.c_str(), "VmHWM: %zu kB", &kb);
            return kb;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    init_alphabet();

    run_all_tests();
    if (g_failures > 0) return 1;

    string proteome_path = argc > 1 ? argv[1] : "proteome.bin";

    cout << "=== real-proteome benchmark ===\n";
    auto t0 = Clock::now();
    Proteome proteome = load_proteome_binary(proteome_path);
    auto t1 = Clock::now();
    cout << proteome.ids.size() << " proteins loaded in "
         << chrono::duration<double>(t1 - t0).count() << "s\n";

    const int k = 9;
    const int max_mismatches = 1;

    size_t rss_before = get_vm_hwm_kb();
    auto b0 = Clock::now();
    KmerIndex idx = build_kmer_index(proteome, k);
    auto b1 = Clock::now();
    size_t rss_after = get_vm_hwm_kb();

    double build_seconds = chrono::duration<double>(b1 - b0).count();
    cout << "\nindex build (one-time): " << build_seconds << "s  ("
         << idx.unique_keys.size() << " distinct k-mers, "
         << idx.positions.size() << " total positions)\n";
    cout << "peak RSS before build: " << rss_before / 1024.0 << " MB\n";
    cout << "peak RSS after build:  " << rss_after / 1024.0 << " MB\n";
    cout << "approx memory for index: " << (rss_after - rss_before) / 1024.0 << " MB\n\n";

    vector<string> queries = {"RDLEAEHVLP", "MELSAEYLX", "MELSAEYLR"};
    for (auto& q : queries) {
        auto q0 = Clock::now();
        vector<MatchResult> matches;
        for (int i = 0; i + k <= (int)q.size(); ++i) {
            string fragment = q.substr(i, k);
            auto m = find_all_with_mismatch(fragment, idx, proteome, max_mismatches);
            for (auto& r : m) matches.push_back(move(r));
        }
        auto q1 = Clock::now();
        double ms = chrono::duration<double, milli>(q1 - q0).count();
        cout << "query " << q << " -> " << matches.size() << " hits in " << ms << " ms\n";
    }

    return 0;
}
