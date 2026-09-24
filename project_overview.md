# Project Overview

## Why this repo exists

When someone designs a peptide drug, they need to check it won't provoke a harmful immune reaction before it goes near a human. There are two independent safety checks:

**1. Similarity to the human proteome.**
If a candidate peptide closely resembles an existing human protein sequence, the immune system is more likely to already tolerate it. If it doesn't resemble anything human, it needs check #2. This is a pure string-matching problem — no ML model required, just an exact/near-match search of the peptide against a reference human proteome. See the "Unrelated side task" row below.

**2. Immunogenicity via HLA/MHC presentation.**
Peptides get displayed to the immune system by **HLA molecules** — the "ID badge" proteins on human cells. If a peptide binds strongly to HLA alleles that are common in real populations, T-cells may recognize it as foreign, which is bad for drug safety. Predicting whether a *given* peptide binds a *given* HLA allele is done by existing open-source models, not built here. What this repo *does* build is the **data layer**: clean, population-representative lists of which HLA alleles are actually common in humans, broken down by ancestry/population, so an immunogenicity check is weighted toward the alleles people actually carry rather than tested against every rare allele indiscriminately.

That data layer is what eventually feeds a **deimmunization loop**: screen a binder against population-weighted HLA alleles → if a window looks risky, mutate it → re-screen → repeat until safe → finally verify the mutated peptide still binds its original target (via AlphaFold/Boltz-2), since a mutation that fixes immunogenicity but breaks binding is useless.

## What's actually in this repo

| Kind | Files | Purpose |
|---|---|---|
| Raw data | [`afnd.tsv`](afnd.tsv) | Allele Frequency Net Database (AFND) export — allele X at frequency Y in population Z, sample size N |
| Library | [`utils.py`](utils.py) | Cleaning/collapsing functions used by the pipeline |
| Exploration | `EDA.ipynb`, `EDA2.ipynb`, `EDA3.ipynb`, `EDA-diff-resolutions.ipynb` | Discovery of data problems (inconsistent resolution, naming systems, tiny studies) and fixes |
| Pipeline run (class I) | `test_pipeline.ipynb` → `4_digit.csv` → `cleaned_data_normal.csv` | Cleaned HLA-A/B/C dataset, all alleles normalized to 4-digit resolution |
| Pipeline run (class II) | `HLA-class2.ipynb` → `4_digit_class2.csv` → `cleaned_data_normal_class2.csv` | Same pipeline for DRB1/DQA1/DQB1/DPA1/DPB1 |
| **Main outputs** | `find_most_common.ipynb` → [`top_100_class1.csv`](top_100_class1.csv), [`top_100_class2.csv`](top_100_class2.csv) | Top 100 alleles per study/population, for class I and class II |
| **Main output** | [`bingsong_output/nmdp_population_sizes.csv`](bingsong_output/nmdp_population_sizes.csv) | Census/ACS-derived population size estimate for each of the 21 U.S. NMDP donor-registry groups, so HLA frequency results can be weighted by how many people each group actually represents — see [NMDP_POPULATION_SIZES.md](NMDP_POPULATION_SIZES.md) for the full derivation |
| Side task | `bingsong_hla_task.ipynb` → the other CSVs in [`bingsong_output/`](bingsong_output/) | Per-population class I frequency files for those same 21 NMDP populations, requested by collaborator Bing Song |
| Unrelated side task | `task3.ipynb`, `task3-pytest-hashtable.py`, `task3-pytest-bruteforce.py`, `uniprotkb_human_ref_proteome_dict.pkl` | Peptide k-mer matching against the human proteome — this is check #1 above, done as a string-matching exercise (hashtable vs. brute-force comparison) |
| Other | `hla_embeddings_all.pkl`, `top_100.csv`, `venvName/` | Allele embedding lookup, an older top-100 export, Python virtualenv |

## The three main deliverables, in more detail

- **[`top_100_class1.csv`](top_100_class1.csv)** / **[`top_100_class2.csv`](top_100_class2.csv)**: the highest-frequency HLA alleles per well-sampled global population study, normalized to a common 4-digit resolution so frequencies are comparable across studies that originally reported at different levels of detail. Full derivation and caveats in [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md).
- **[`nmdp_population_sizes.csv`](bingsong_output/nmdp_population_sizes.csv)**: a real-world U.S. population size estimate for each of the **21 NMDP donor-registry groups**, built from Census/ACS tables and IPUMS microdata cross-tabs. See [NMDP_POPULATION_SIZES.md](NMDP_POPULATION_SIZES.md) for how each of the 21 group sizes was derived, including the judgment calls involved.

This file happens to sit in the same [`bingsong_output/`](bingsong_output/) folder as the per-population class I HLA frequency CSVs, but the two are **not related** — the frequency CSVs are a separate, one-off side task for collaborator Bing Song (`bingsong_hla_task.ipynb`), built from HLA study data with no code path connecting them to the Census-derived population sizes. They just share a folder and the same 21 group labels.

## Your role in the larger pipeline

You are not building the binding predictor, the mutation loop, or the structure-verification step. Your work — cleaning AFND data, normalizing resolutions, computing top-100 allele lists, and sizing the NMDP population groups — provides the population-representative HLA ground truth that the immunogenicity-checking engine and the deimmunization loop consume as input.
