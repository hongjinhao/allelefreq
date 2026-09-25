# NMDP Population Group Sizing: Overview and Deep Dive

## Part 1: The short version

### What is this document about?

This explains where the numbers in [`nmdp_population_sizes.csv`](nmdp_population_sizes.csv) come from, and why the 21 NMDP groups are split the way they are.

Note: this file previously lived in the same `bingsong_output/` folder as the per-population HLA class I frequency CSVs (now under `bingsong_side_task/`), but it is **not derived from or joined with them** — nothing in `bingsong_hla_task.ipynb` reads this file. The frequency CSVs come from HLA study data (a side task for collaborator Bing Song); this file is an independent Census/ACS demographic estimate of how many people belong to each of the 21 NMDP groups. They happened to share the same 21 group labels and, at one point, the same output folder, but are otherwise unrelated pieces of work.

### The problem being solved

The **NMDP (National Marrow Donor Program)**, which runs the U.S. bone marrow donor registry, groups its donors into **21 detailed population subgroups** (e.g. "African American pop 2," "South Asian Indian," "Caribbean Hispanic"). These groups exist so that HLA haplotype frequencies estimated within each group are meaningful — you can only estimate reliable allele frequencies for a group if:

1. **Genetic similarity** — the people in the group are similar enough that one frequency table describes them reasonably well, and
2. **Sample size adequacy** — there are enough people in the group for the frequency estimate to be statistically stable.

Donors self-report race/ethnicity in a much coarser way than these 21 groups, so the NMDP subdivides (and sometimes merges) categories to satisfy both criteria — e.g. "Black or African American" is split four ways by migration history (U.S.-born vs. sub-Saharan African-born vs. via Central/South America vs. via the Caribbean), because those subpopulations have measurably different HLA allele distributions despite sharing a single Census race category.

The task here is: **for each of the 21 NMDP groups, estimate how many people in the U.S. actually belong to it**, using public Census/ACS data. That population-size estimate is what [`nmdp_population_sizes.csv`](nmdp_population_sizes.csv) records — a standalone demographic reference, independent of the HLA frequency work in `bingsong_side_task/`.

### Data sources used

- **Census pre-aggregated tables** (e.g. `B02001` Race Alone) — fast, but only give the categories the Census Bureau chose to publish.
- **IPUMS / ACS PUMS** (Public Use Microdata Samples) — the raw underlying survey microdata, queried via Berkeley's SDA online tool, used whenever a pre-made table didn't have the exact cross-tabulation needed (e.g. race × birthplace, or race × ancestry).
- **5-year ACS estimates** were preferred over 1-year estimates throughout, because pooling 5 years of survey responses gives a larger, more statistically reliable sample — important for small subpopulations like "Caribbean Indian" (~57,000 people nationally) where a 1-year sample would be too noisy.

### The result: 21 groups, 5 broad categories

| Broad Census race/ethnicity | NMDP subgroups | Estimated size |
|---|---|---|
| Black or African American | African American pop 2, African, Black South or Central American, Caribbean Black | 38.2m / 2.31m / 0.197m / 2.05m |
| Asian | Chinese, Filipino, Japanese, Korean, South Asian Indian, Southeast Asian, Vietnamese | 4.55m / 3.04m / 0.75m / 1.50m / 5.78m / 1.34m / 1.92m |
| Hispanic or Latino | Mexican or Chicano, Hispanic South or Central American, Caribbean Hispanic | 37.9m / 11.7m / 11.0m |
| White | European Caucasian, Middle Eastern or North Coast of Africa (MENA) | 192.2m / 3.92m |
| Native American | North American Amerindian, Alaska Native or Aleut, American Indian South or Central America | 1.74m / 0.12m / 0.42m |
| Native Hawaiian or Other Pacific Islander | Hawaiian or other Pacific Islander, Caribbean Indian | 0.63m / 0.057m |

The full table with exact figures is in [`nmdp_population_sizes.csv`](nmdp_population_sizes.csv).

---

## Part 2: Deep dive — how each number was built

The core technique throughout is: **take the total for a Census race category, then subtract or filter down to isolate the sub-population the NMDP group actually describes.** Three general strategies show up repeatedly:

1. **Subtraction** — take a broad category and remove a piece that belongs to a different NMDP group (e.g. all Black Americans minus foreign-born Black Americans = "African American pop 2").
2. **Cross-tabulation (race × birthplace or race × ancestry)** — when no single pre-made table isolates the group, query IPUMS PUMS directly for the intersection (e.g. "people who selected race=Black AND were born in the Caribbean").
3. **Ancestry-based reassignment** — for groups defined by ethnicity rather than Census race category (MENA, Armenian, Turkish), use the ancestry table (`B04006`) instead of the race table, since these groups don't map cleanly onto Census race options and would otherwise be invisible or miscounted as "White."

### Black or African American (4 subgroups)
- **African American pop 2** (U.S.-born Black Americans): Total Black alone (`B02001`, 40.92m) minus foreign-born Black (`B05006`, 2.763m) = **38.2m**.
- **African**: Foreign-born Black from Africa (2.763m) minus foreign-born Black from Northern Africa (0.454m, who are genetically/culturally distinct — closer to MENA) = **2.31m**.
- **Black South or Central American**: sum of Black immigrants from Central America, South America, and Mexico via IPUMS race × birthplace cross-tab = **0.197m**.
- **Caribbean Black**: Black immigrants from Cuba + West Indies via the same cross-tab = **2.05m**.

This is the clearest example of the migration-history logic: all four groups share the Census "Black" race category, but the NMDP splits them by the geographic path each subpopulation's ancestors took, because HLA allele frequencies differ by that history, not just by self-reported race.

### Asian (7 subgroups)
Built almost entirely from `B02015` (Asian Alone by Selected Groups), which already breaks Asian identity into detailed ethnic groups — so this category needed the least manual reconstruction. A few groups are sums of related sub-ethnicities:
- **South Asian Indian** = Asian Indian + Nepalese + Bangladeshi + Pakistani + Sri Lankan + Bhutanese = 5.78m.
- **Southeast Asian** = Hmong + Cambodian + Thai + Laotian + Burmese + Indonesian + Malaysian + Singaporean + Mien = 1.34m.
- Chinese, Filipino, Japanese, Korean, Vietnamese are each closer to a direct table lookup (with Taiwanese folded into Chinese, and Okinawan folded into Japanese).

### Hispanic or Latino (3 subgroups)
Built from `B03001` (Hispanic or Latino Origin by Specific Origin):
- **Mexican or Chicano** = Mexican alone = 37.92m.
- **Hispanic South or Central American** = sum of all Central + South American countries (including Panama, deliberately — see judgment call below) = 11.69m.
- **Caribbean Hispanic** = Puerto Rican + Cuban + Dominican = 10.98m.
- "Other Hispanic or Latino" (Spaniard, Spanish, Spanish American, and unspecified — 4.17m combined) is **excluded entirely**, since it doesn't map to a genetically coherent NMDP group.

**Judgment call — Panama**: Panama's population is included under Central/South American Hispanic rather than Caribbean, on the reasoning that Panama's substantial Afro-Panamanian population is more likely to self-identify as Black in Census data anyway, making a Caribbean-Hispanic assignment for the remaining Panamanian respondents less genetically ambiguous than it first appears.

### White (2 subgroups)
- **European Caucasian**: White alone from `B03002` = 192.2m (the largest single group by a wide margin).
- **Middle Eastern or North Coast of Africa (MENA)**: Built from `B04006` (ancestry, not race) since MENA people are not a distinct Census race option and are usually recorded as "White." Sum of Arab + Iranian + Afghan + Armenian + Assyrian/Chaldean/Syriac + Israeli + Turkish = 3.92m, despite being administratively "White," because HLA profiles for MENA populations are measurably distinct from European Caucasian ones.

**Judgment calls flagged in the source:**
- *Armenian and Turkish* are included in MENA "but debatable" — geographically/culturally these straddle Europe and the Middle East, but their HLA profiles are described as closer to Middle Eastern populations, which is why they're kept in this group.
- *Cypriot* (10,843) was considered but excluded — negligible either way.
- *Known undercount*: ancestry questions allow multiple answers (double-counting risk) and ~80 million respondents didn't report an ancestry at all, so the true MENA population is likely somewhat higher than 3.9m.

### Native American (3 subgroups)
Built from `B02014` (American Indian and Alaska Native by Selected Tribal Groupings) plus an IPUMS cross-tab:
- **North American Amerindian**: total AIAN tribal counts minus the Central/Mexican/South American Indian subtotal (506,517) that belongs to a different NMDP group = 1.74m.
- **Alaska Native or Aleut**: specified + unspecified Alaska Native tribes = 0.12m.
- **American Indian South or Central America**: foreign-born-only estimate via IPUMS race × birthplace (Mexico + Central America + South America) = 0.416m.

**Two flagged limitations:**
1. The South/Central American Indian figure only captures the **foreign-born** — U.S.-born children of these immigrants show up as domestic AIAN and get folded into "North American Amerindian" instead, with no clean way to separate them.
2. Mexico alone is 64% of that group's total, raising the same "which bucket does Mexican-origin indigenous ancestry belong in" judgment call seen with Panama — if Mexico were moved to the North American group instead, the South/Central American figure would drop to ~149,000.

### Native Hawaiian or Other Pacific Islander (2 subgroups)
- **Hawaiian or other Pacific Islander**: direct total from `B02016` = 0.63m.
- **Caribbean Indian** (South Asian-descended people who settled in the Caribbean, e.g. via 19th-century indentured labor migration): the hardest group to size, built through a 4-step IPUMS filtering process:
  1. Race=Other Asian/Pacific Islander × Caribbean birthplace → ~32,000 (foreign-born only, undercounts U.S.-born generations).
  2. Race=Other Asian/Pacific Islander × Caribbean primary ancestry (ANCESTR1) → ~47,000, which **superseded** (not added to) step 1 since it captures both foreign- and U.S.-born.
  3. Coverage check: filtered "Asian Indian" ancestry across *all* race codes to see how many fall outside the "Other Asian/Pacific Islander" bucket — found 5.9% do, so applied a ~6% upward correction (+~3,200).
  4. Added Caribbean ancestry reported as a person's *secondary* ancestry (ANCESTR2) — a clean additive step since ANCESTR1/ANCESTR2 are mutually exclusive fields → +~6,736.
  - Final: **~57,000**, flagged as still a likely undercount since third-generation-plus descendants who no longer report Caribbean ancestry are invisible to this method.

### Recurring theme: judgment calls are documented, not hidden

Nearly every group required at least one non-obvious inclusion/exclusion decision (Panama, Mexico, Armenian/Turkish, Cypriot, "Other Hispanic"). In each case the source reasoning is preserved above rather than presented as a clean number, because these estimates feed into how HLA frequency data gets weighted or compared across groups — a downstream user needs to know which boundary decisions were made and why, in case a different NMDP-alignment choice would change the analysis.
