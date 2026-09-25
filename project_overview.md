# Project Overview

## Why this repo exists

When someone designs a peptide drug, they need to check it won't provoke a harmful immune reaction before it goes near a human. There are two independent safety checks:

**1. Similarity to the human proteome.**
If a candidate peptide closely resembles an existing human protein sequence, the immune system is more likely to already tolerate it. If it doesn't resemble anything human, it needs check #2. This is a pure string-matching problem — no ML model required, just an exact/near-match search of the peptide against a reference human proteome. See [`task3_string_matching/`](task3_string_matching) below.

**2. Immunogenicity via HLA/MHC presentation.**
Peptides get displayed to the immune system by **HLA molecules** — the "ID badge" proteins on human cells. If a peptide binds strongly to HLA alleles that are common in real populations, T-cells may recognize it as foreign, which is bad for drug safety. Predicting whether a *given* peptide binds a *given* HLA allele is done by existing open-source models, not built here. What this repo *does* build is the **data layer**: clean, population-representative lists of which HLA alleles are actually common in humans, broken down by ancestry/population, so an immunogenicity check is weighted toward the alleles people actually carry rather than tested against every rare allele indiscriminately.

That data layer is what eventually feeds a **deimmunization loop**: screen a binder against population-weighted HLA alleles → if a window looks risky, mutate it → re-screen → repeat until safe → finally verify the mutated peptide still binds its original target (via AlphaFold/Boltz-2), since a mutation that fixes immunogenicity but breaks binding is useless.

## Repo layout

The repo is split into one top-level folder per task. Each task with real depth has its own overview doc — read that first for anything beyond a skim.

| Folder | Task | Overview doc |
|---|---|---|
| [`allelefreq_analysis/`](allelefreq_analysis) | Main pipeline: clean AFND data → normalize allele resolution → top-100 alleles per population, for HLA class I and class II. This is check #2's data layer. | [ANALYSIS_OVERVIEW.md](allelefreq_analysis/ANALYSIS_OVERVIEW.md) |
| [`population_size/`](population_size) | How many real people (US NMDP donor-registry members) fall into each of the 21 population groups the analysis above reports on, so results can be population-weighted. Independent of the analysis pipeline — consumes only the group labels, not its outputs. | [NMDP_POPULATION_SIZES.md](population_size/NMDP_POPULATION_SIZES.md) (derivation) + [POPULATION_COVERAGE.md](population_size/POPULATION_COVERAGE.md) (cross-check against `allelefreq_analysis` outputs) |
| [`task3_string_matching/`](task3_string_matching) | Peptide-vs-human-proteome k-mer matching (check #1 above) — a self-contained algorithms exercise, brute-force vs. hashtable-indexed vs. a C++ port, unrelated to the HLA pipeline. | [TASK3_STRING_MATCHING.md](task3_string_matching/TASK3_STRING_MATCHING.md) |
| [`bingsong_side_task/`](bingsong_side_task) | One-off per-population class I frequency export requested by collaborator Bing Song, using a different (unnormalized) collapse rule and a fixed allele vocabulary from `hla_embeddings_all.pkl`. No dedicated doc — documented as §2.7 of [ANALYSIS_OVERVIEW.md](allelefreq_analysis/ANALYSIS_OVERVIEW.md). | — |
| [`archive/`](archive) | `top_100.csv`, a pre-class-split export superseded by `allelefreq_analysis/top_100_class1.csv` / `top_100_class2.csv`. Kept for reference only, nothing reads it. | — |

## Your role in the larger pipeline

You are not building the binding predictor, the mutation loop, or the structure-verification step. Your work — cleaning AFND data, normalizing resolutions, computing top-100 allele lists, and sizing the NMDP population groups — provides the population-representative HLA ground truth that the immunogenicity-checking engine and the deimmunization loop consume as input.
