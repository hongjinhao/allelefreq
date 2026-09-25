# Task 3: Peptide String Matching — Overview and Deep Dive

## Part 1: The short version

### What is this task?

This is the **"unrelated side task"** noted in [project_overview.md](../project_overview.md) — separate from the main HLA allele-frequency pipeline in this repo. It's a self-contained algorithms exercise: given a candidate drug peptide, quickly check whether it closely resembles any existing human protein.

That check matters biologically (per [project_overview.md](../project_overview.md)): if a peptide drug closely resembles a human protein, the immune system likely already tolerates it. If it doesn't resemble anything human, it needs the second, separate safety check this repo's main pipeline feeds into — HLA-binding immunogenicity risk. This task only covers the first check, done as a pure string-matching problem, no ML involved.

**The problem, formally:**
- **String A** — a query peptide (e.g. `"RDLEAEHVLP"`)
- **String B** — the human reference proteome, ~83,000 protein sequences
- Break the query into overlapping **k-mers** (`k=9` by default)
- Find every place each k-mer occurs in the proteome, allowing up to `max_mismatches` (default `1`) mismatched residues

### Files

| File | Purpose |
|---|---|
| [task3.ipynb](task3.ipynb) | Original spec/scratch notebook — worked example, algorithm description, stubbed-out function |
| [task3-pytest-bruteforce.py](task3-pytest-bruteforce.py) | Brute-force implementation + 8 pytest cases |
| [task3-pytest-hashtable.py](task3-pytest-hashtable.py) | Hashtable-indexed implementation, same 8 pytest cases (now actually different code — see history note below) |
| [task3_benchmark.py](task3_benchmark.py) | Standalone script to time both implementations against the real proteome |
| [uniprotkb_human_ref_proteome_dict.pkl](uniprotkb_human_ref_proteome_dict.pkl) | The proteome itself: `{protein_id: sequence}`, 83,413 proteins |
| [query_peptides.csv](query_peptides.csv) | 23 realistic query peptides (real drugs/hormones/research peptides + validation controls) for exercising the matcher beyond the toy pytest strings |
| [query_sequences.py](query_sequences.py) | Loads `query_peptides.csv` into the `query_sequences` list shape the matcher expects |
| [cpp_prototype/task3_hashtable.cpp](cpp_prototype/task3_hashtable.cpp) | C++ port of the hashtable algorithm using a flat CSR index instead of a dict-of-lists — see §2.6 |
| [cpp_prototype/export_proteome.py](cpp_prototype/export_proteome.py) | One-time export of the proteome pickle to a flat binary file the C++ prototype reads directly |
| [cpp_prototype/proteome.bin](cpp_prototype/proteome.bin) | The exported binary proteome (32 MB) |

**History note:** originally, `task3-pytest-hashtable.py` was a byte-for-byte copy of the brute-force file — the hashtable version referenced by its filename had never actually been written. It now contains a real hashtable-indexed implementation (see Part 2).

### The two algorithms, at a glance

| | Brute force | Hashtable-indexed |
|---|---|---|
| Core idea | Slide each query k-mer across every position of every protein, count mismatches | Index every proteome k-mer once; at query time, look up all mismatch-tolerant variants of the query k-mer |
| Time per query | O(g · m · k) — rescans the whole proteome every time | O(V · k) average-case — independent of proteome size, `V` ≈ 172 for k=9, max_mismatches=1 |
| One-time cost | none | O(p · k) to build the index (`p` = total proteome length) |
| Space | O(m) per protein scanned | O(p) for the index |
| Measured (real proteome, k=9, mismatches≤1) | ~1–2 seconds **per query**, capped to a 2,000-protein subset (full 83,413 would take much longer) | 24.8s index build (once) + **0.1–0.5 ms per query** after that |

Roughly **4,500–8,000× faster per query**, even measuring brute force against a proteome subset 42× smaller than the real one.

The hashtable algorithm above has two implementations: the Python one used for that comparison, and a C++ port that keeps the same algorithm but uses a flat, cache-friendly memory layout instead of Python's dict-of-lists — **~7× faster to build the index and ~4.6× less memory**, correctness-verified against the same 8 pytest cases. See §2.6.

---

## Part 2: Deep dive

### 2.1 The brute-force algorithm

Two nested layers, in [task3-pytest-bruteforce.py](task3-pytest-bruteforce.py):

**`find_all_with_mismatch(fragment, sequence, protein_id, max_mismatches)`** — slides a window of length `k` across one protein sequence, comparing character-by-character:
```python
for pos in range(len(sequence) - frag_len + 1):
    candidate = sequence[pos:pos + frag_len]
    mismatch_list = [(i, fragment[i], candidate[i]) for i in range(frag_len) if fragment[i] != candidate[i]]
    if len(mismatch_list) <= max_mismatches:
        matches.append({...})
```
Cost: O(m·k) per (fragment, protein) pair, `m` = protein length.

**`find_peptide_overlaps(query_sequences, proteome_dict, k, max_mismatches)`** — for every query, for every one of the 83,413 proteins, for every k-mer fragment of that query, runs the scan above:
```python
for A in query_sequences:                    # h queries
    for B_id, B in proteome_dict.items():     # g = 83,413 proteins
        for i in range(len(A) - k + 1):       # n-k+1 fragments
            matches = find_all_with_mismatch(fragment, B, ...)
```
Total: **O(h · g · n · m · k)**. The file's own comment calls this "roughly 10^12" operations at real proteome scale. The core problem: every query rescans the entire proteome from nothing, even though the proteome never changes between queries.

### 2.2 The hashtable-indexed algorithm

Implemented in [task3-pytest-hashtable.py](task3-pytest-hashtable.py). Key idea: since the proteome is static across queries, index it **once**, then make each query a small number of O(1) lookups instead of a full rescan.

**Step 1 — build the index once** ([`build_kmer_index`](task3-pytest-hashtable.py)):
```python
index = defaultdict(list)
for protein_id, sequence in proteome_dict.items():
    for pos in range(len(sequence) - k + 1):
        index[sequence[pos:pos + k]].append((protein_id, pos))
```
Every exact k-mer that occurs anywhere in the proteome maps to every `(protein_id, position)` where it occurs. Cost: O(p·k), `p` = total proteome length — paid once, not once per query.

**Step 2 — generate mismatch-tolerant neighbors of the *query* k-mer** ([`generate_mismatch_neighbors`](task3-pytest-hashtable.py)):

For a 9-residue fragment and `max_mismatches=1`, this yields every string within Hamming distance 1: the fragment itself, plus each of its 9 positions substituted with each of the other 19 amino acids — **172 total variants**.
```python
fragment = "RDLEAEHVL"
neighbors = generate_mismatch_neighbors(fragment, max_mismatches=1)
# -> ['RDLEAEHVL', 'ADLEAEHVL', 'CDLEAEHVL', ..., 'RDLEAEHVA']  (172 strings)
```
Implemented generally via `itertools.combinations` (which positions to mutate) × `itertools.product` (what to mutate them to), so it also works for `max_mismatches=2` and beyond — see §2.4 for why that stops being practical fast.

**Step 3 — look up every neighbor, once each, in the exact-match index** ([`find_all_with_mismatch`](task3-pytest-hashtable.py)):
```python
for variant in generate_mismatch_neighbors(fragment, max_mismatches):
    for protein_id, pos in index.get(variant, []):
        # any hit here is automatically within budget — the variant differs
        # from the original fragment by <= max_mismatches by construction
```
Most of the 172 lookups return nothing (a random 9-mer almost never occurs in the proteome); any that do hit are guaranteed valid matches, no further scanning needed.

**Why this is the right way around:** the mismatch tolerance is pushed onto the *query* side (small: one peptide, a handful of k-mers) rather than the *proteome* side (huge: 83,413 proteins). Indexing the proteome side with pre-expanded neighbors instead — i.e. storing all 172 mismatch variants of every proteome k-mer as index keys — was tried conceptually and rejected; see §2.4.

**Complexity comparison:**

| | Brute force | Hashtable-indexed |
|---|---|---|
| Build (one-time) | — | O(p·k) |
| Per query | O(g · n · m · k) | O(n · V · k), `V` ≈ `k·19+1` for max_mismatches=1 |
| Proteome-size dependence | scales with `g`, `m` every query | none, after the index is built |

### 2.3 Measured results (real proteome: 83,413 proteins, k=9, max_mismatches=1)

Run via [task3_benchmark.py](task3_benchmark.py):

```bash
python3 task3_benchmark.py --include-brute-force --brute-force-limit 2000
```

| Query | Hashtable (full 83,413 proteins) | Brute force (2,000-protein subset only) |
|---|---|---|
| `RDLEAEHVLP` | 0.49 ms | 2,237 ms |
| `MELSAEYLX` | 0.14 ms | 956 ms |
| `MELSAEYLR` | 0.14 ms | 1,147 ms |

Index build (one-time, amortized across every query that follows): **24.8s**.

Note the brute-force numbers are against a proteome **42× smaller** than the real one — full-proteome brute force would be far slower still (extrapolating, tens of seconds to minutes per query). The comparison in the table already understates the real gap.

### 2.4 Memory: when does a hashtable *actually* stop working?

A researcher's comment that "hashtable doesn't work, too much space" is legitimate for some designs of this problem — it's about *how* you index, not whether indexing works at all. Measured directly:

| Index design | Keys | Approx. memory |
|---|---|---|
| **This implementation (Python)** — exact proteome k-mers only, neighbors generated per-query | 11,443,421 distinct k-mers (30,444,422 total position entries) | **~4.7 GB** (measured via `resource.getrusage`) |
| **Same algorithm, C++ flat CSR index** — same keys, packed into contiguous arrays instead of a dict-of-lists | same 11,443,421 / 30,444,422 | **~1.0 GB** (measured via `/proc/self/status` peak RSS) — see §2.6 |
| **The "wrong way around"** — expand every proteome k-mer to its 172 mismatch-neighbors and index those | 30,444,422 × 172 ≈ **5.24 billion** keys | Would be in the **terabyte range** — infeasible on any single machine |

4.7 GB is workable on a normal dev machine but isn't free — Python dict overhead is substantial (~150 bytes per entry here, for what's structurally a 9-character string and a short list). A few realistic ways this stops fitting:

1. **Indexing the proteome side with pre-expanded neighbors** instead of the query side (the ~460× blowup shown above) — the most likely single explanation for a flat "doesn't work" verdict.
2. **`max_mismatches` beyond 1.** Neighbor count is `Σ_{i=0}^{m} C(k,i)·19^i`: 172 at m=1, ~13,000 at m=2, ~800,000 at m=3 for k=9. Query-side generation stays fine through m=2; proteome-side expansion at m=2 would be catastrophic.
3. **A larger reference database.** This task only indexes the human reference proteome. Real deimmunization workflows sometimes check against multi-species panels or all isoforms — scale `p` up 10–100× and a naive in-memory Python dict index stops fitting in RAM.
4. **A different structure entirely** — e.g. a general suffix trie/tree to support arbitrary-length queries rather than fixed-`k` k-mers. Naive tries have per-node/per-character overhead and are a well-known way to blow up memory on genome/proteome-scale text, which real tools address with compressed structures (FM-index/BWT, suffix arrays, or on-disk/sketch-based indices like those in BLAST, DIAMOND, MMseqs2, Jellyfish).

### 2.5 Test data

The pytest suites (both files) only ever exercise a 1-protein, 45-residue toy fixture — enough to check correctness in milliseconds, not to exercise real-world scale or realistic peptide sequences. [query_peptides.csv](query_peptides.csv) fills that gap with 23 entries:

- **20 real peptides**: therapeutic/hormone peptides spanning diabetes drugs (insulin, glucagon, GLP-1, exenatide, pramlintide), hormones (oxytocin, vasopressin, calcitonin, teriparatide), an HIV fusion inhibitor (enfuvirtide), an anticoagulant (bivalirudin), and antimicrobial/research peptides (LL-37, melittin, magainin 2). Descriptions note where a real drug carries modifications (D-amino acids, cyclization, lipidation) that a plain 20-letter string can't represent.
- **3 synthetic controls**, useful for validating any matcher implementation: an exact-match control and a 1-mismatch control, both cut directly from real entries in `uniprotkb_human_ref_proteome_dict.pkl` (guaranteed to hit), and a scrambled negative control (should never hit).
- Angiotensin II (8 residues) is deliberately shorter than `k=9`, to exercise that edge case with a real peptide rather than a synthetic placeholder.

[query_sequences.py](query_sequences.py) loads the CSV into the list shape `find_peptide_overlaps` expects, plus a metadata lookup by sequence.

### 2.6 The C++ prototype: same algorithm, no per-object boxing

Python's dict-of-lists index works (§2.3), but every entry it stores is a chain of separately heap-allocated objects — a `str` key, a `list` value, `tuple`s inside it, boxed `int`s inside those. [cpp_prototype/task3_hashtable.cpp](cpp_prototype/task3_hashtable.cpp) implements the *identical algorithm* — same neighbor generation, same lookup logic — over a flat, contiguous memory layout instead, to see how much of the Python cost was the algorithm versus the data structure.

**Design:**
- **Bit-packed k-mer keys.** Each 9-residue k-mer is packed into a single `uint64_t`, 5 bits/residue, covering the real 22-letter alphabet found in the proteome (20 standard residues + ambiguity codes `U`, `X`). Comparing two k-mers becomes one integer comparison instead of a 9-byte string compare behind a pointer.
- **CSR-style flat index** instead of a dict-of-lists: a sorted `unique_keys` array, a parallel `offsets` array, and one contiguous `positions` array holding every `(protein_idx, pos)` pair grouped by key. Reading all matches for a key is one slice of one array — no per-key heap allocation.
- **A flat open-addressing table** (linear probing, ~0.7 load factor) maps `key -> row index` into that CSR structure for O(1) average lookup, instead of `std::unordered_map`'s node-based chaining (which would just reintroduce per-entry heap allocations).
- Mismatch-neighbor generation mirrors the Python version exactly — `itertools.combinations`/`product` become nested loops over chosen positions and substitution letters — so variant counts match precisely (172 for k=9, max_mismatches=1).

**Correctness first:** the same 8 pytest cases from [task3-pytest-hashtable.py](task3-pytest-hashtable.py) are ported directly into the C++ file and run before any benchmark — all 8 pass, and on the real proteome the C++ version returns the exact same hit counts as the Python version (9, 7, 7 for the three benchmark queries), cross-validating both implementations independently.

**Measured results** (same machine, same real proteome, k=9, max_mismatches=1):

| | Python (dict-of-lists) | C++ (flat CSR + open addressing) | Improvement |
|---|---|---|---|
| Index build (one-time) | 24.8–26.1s | **3.49s** | ~7.3× faster |
| Peak memory for the index | 4,672 MB | **1,020 MB** | ~4.6× less |
| Query `RDLEAEHVLP` | 0.490 ms | **0.056 ms** | ~8.8× faster |
| Query `MELSAEYLX` | 0.144 ms | **0.026 ms** | ~5.5× faster |
| Query `MELSAEYLR` | 0.142 ms | **0.022 ms** | ~6.5× faster |

**Why 1,020 MB and not the ~576 MB component estimate:** that lower figure only accounts for the final steady-state structures. *Peak* RSS also captures a transient buffer — the unsorted `vector<pair<uint64_t, PositionEntry>>` used before collapsing into CSR form (30.4M × 16 bytes ≈ 498 MB) — which is still resident while the final arrays are being built alongside it, plus some `std::vector` growth over-allocation from unreserved `push_back` calls. Freeing that buffer immediately after the CSR collapse and pre-reserving the output arrays would likely cut the peak further, to somewhere around 650–700 MB.

**Reproduce:**
```bash
cd cpp_prototype
python3 export_proteome.py                                    # one-time: pickle -> proteome.bin
g++ -O2 -std=c++17 -o task3_hashtable task3_hashtable.cpp
./task3_hashtable proteome.bin
```

### 2.7 Open items / current limitations

- **No real candidate query peptides exist for this pipeline yet.** Every `query_sequences` value anywhere in the repo, before `query_peptides.csv`, was a toy string from the notebook's worked example or pytest fixtures — there's no evidence this check has ever run against an actual drug-candidate peptide.
- **`max_mismatches` is only exercised at 1** in the pytest suite; the neighbor-generation code (both Python and C++) supports higher values but nothing here has validated performance or correctness at `max_mismatches=2+`.
- **This task isn't wired into the rest of the repo.** It shares no code path with the HLA allele-frequency pipeline that's the repo's main output — it's a standalone proof-of-concept for one of the two safety checks described in [project_overview.md](../project_overview.md), not an integrated pipeline stage.
- **The C++ prototype is standalone**, per its own design goal — it reads a manually-exported binary snapshot of the proteome rather than being callable from the notebooks, and its memory peak hasn't been tightened past the transient-buffer issue noted in §2.6.
