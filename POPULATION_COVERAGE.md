# Population Coverage: Class I vs. Class II Output

This documents which populations appear in [`top_100_class1.csv`](top_100_class1.csv) and [`top_100_class2.csv`](top_100_class2.csv), split into:

1. NMDP-related populations present in the output
2. NMDP populations **missing** from the output (compared against the 21 official groups in [`bingsong_output/nmdp_population_sizes.csv`](bingsong_output/nmdp_population_sizes.csv), documented in [`NMDP_POPULATION_SIZES.md`](NMDP_POPULATION_SIZES.md))
3. Non-NMDP populations present in the output

**Note on spelling:** the output CSVs consistently spell the three Caribbean NMDP groups as "Caribean" (missing a "b"), while the reference file spells them "Caribbean". These are treated as the same group below (e.g. `USA NMDP Caribean Black` = `USA NMDP Caribbean Black`).

---

## 1. NMDP-related populations present

Both Class I and Class II contain the same 20 of the 21 official NMDP groups (population names below use the spelling as they appear in the output CSVs).

| # | NMDP group (official name) | In Class I | In Class II |
|---|---|---|---|
| 1 | African American pop 2 | ✅ `USA NMDP African American pop 2` | ✅ `USA NMDP African American pop 2` |
| 2 | African | ✅ `USA NMDP African` | ✅ `USA NMDP African` |
| 3 | Caribbean Black | ✅ `USA NMDP Caribean Black` | ✅ `USA NMDP Caribean Black` |
| 4 | Chinese | ✅ `USA NMDP Chinese` | ✅ `USA NMDP Chinese` |
| 5 | Filipino | ✅ `USA NMDP Filipino` | ✅ `USA NMDP Filipino` |
| 6 | Japanese | ✅ `USA NMDP Japanese` | ✅ `USA NMDP Japanese` |
| 7 | Korean | ✅ `USA NMDP Korean` | ✅ `USA NMDP Korean` |
| 8 | South Asian Indian | ✅ `USA NMDP South Asian Indian` | ✅ `USA NMDP South Asian Indian` |
| 9 | Southeast Asian | ✅ `USA NMDP Southeast Asian` | ✅ `USA NMDP Southeast Asian` |
| 10 | Vietnamese | ✅ `USA NMDP Vietnamese` | ✅ `USA NMDP Vietnamese` |
| 11 | Mexican or Chicano | ✅ `USA NMDP Mexican or Chicano` | ✅ `USA NMDP Mexican or Chicano` |
| 12 | Hispanic South or Central American | ✅ `USA NMDP Hispanic South or Central American` | ✅ `USA NMDP Hispanic South or Central American` |
| 13 | Caribbean Hispanic | ✅ `USA NMDP Caribean Hispanic` | ✅ `USA NMDP Caribean Hispanic` |
| 14 | European Caucasian | ✅ `USA NMDP European Caucasian` | ✅ `USA NMDP European Caucasian` |
| 15 | Middle Eastern or North Coast of Africa | ✅ `USA NMDP Middle Eastern or North Coast of Africa` | ✅ `USA NMDP Middle Eastern or North Coast of Africa` |
| 16 | North American Amerindian | ✅ `USA NMDP North American Amerindian` | ✅ `USA NMDP North American Amerindian` |
| 17 | Alaska Native or Aleut | ✅ `USA NMDP Alaska Native or Aleut` | ✅ `USA NMDP Alaska Native or Aleut` |
| 18 | American Indian South or Central America | ✅ `USA NMDP American Indian South or Central America` | ✅ `USA NMDP American Indian South or Central America` |
| 19 | Hawaiian or other Pacific Islander | ✅ `USA NMDP Hawaiian or other Pacific Islander` | ✅ `USA NMDP Hawaiian or other Pacific Islander` |
| 20 | Caribbean Indian | ✅ `USA NMDP Caribean Indian` | ✅ `USA NMDP Caribean Indian` |

**20 / 21 present in both Class I and Class II.**

---

## 2. Missing NMDP populations

Only one of the 21 official NMDP groups has no corresponding data in either output file:

| # | NMDP group (official name) | In Class I | In Class II |
|---|---|---|---|
| 3 | Black South or Central American | ❌ missing | ❌ missing |

**1 / 21 missing — identical gap in both Class I and Class II.**

---

## 3. Non-NMDP populations present

These are populations in the output that are **not** part of the 21 NMDP groups (e.g. national/registry cohorts, ethnic subgroups sourced from other studies).

### Present in both Class I and Class II (58 populations)

```
China Hubei Han
China Jiangsu Han
China Zhejiang Han
Colombia BogotÃ¡ Cord Blood
Croatia pop 4
Czech Republic NMDR
France French Bone Marrow Donor Registry
Germany DKMS - Austria minority
Germany DKMS - Bosnia and Herzegovina minority
Germany DKMS - China minority
Germany DKMS - Croatia minority
Germany DKMS - France minority
Germany DKMS - German donors
Germany DKMS - Greece minority
Germany DKMS - Italy minority
Germany DKMS - Netherlands minority
Germany DKMS - Portugal minority
Germany DKMS - Romania minority
Germany DKMS - Spain minority
Germany DKMS - Turkey minority
Germany DKMS - United Kingdom minority
Germany pop 6
Germany pop 8
Hong Kong Chinese BMDR
Hong Kong Chinese HKBMDR HLA 11 loci
Hong Kong Chinese cord blood registry
India Tamil Nadu
Israel Arab pop 2
Israel Argentina Jews
Israel Ashkenazi Jews pop 3
Israel Bukhara Jews
Israel Druze
Israel Ethiopia Jews
Israel Georgia Jews
Israel Iran Jews
Israel Iraq Jews
Israel Kavkazi Jews
Israel Libya Jews
Israel Morocco Jews
Israel Poland Jews
Israel Tunisia Jews
Israel USA Jews
Israel USSR Jews
Israel YemenJews
Japan pop 16
Japan pop 3
Netherlands Leiden
Poland BMR
Poland DKMS
Russia Karelia
Russia Nizhny Novgorod, Russians
South Korea pop 10
Spain (Catalunya, Navarra, Extremadura, AaragÃ³n, Cantabria,
USA African American pop 4
USA Arizona Gila River Pima
USA Asian pop 2
USA Caucasian pop 4
USA Hispanic pop 2
```

Note: `Colombia BogotÃ¡ Cord Blood`, `Spain (Catalunya, Navarra, Extremadura, AaragÃ³n, Cantabria,` — these strings contain mojibake (`Ã¡`, `Ã³`) from an encoding mismatch in the source data (should read "Bogotá" and "Aragón"). Reproduced as-is from the CSVs; not corrected here.

### Present in Class I only (2 populations)

```
China South Han pop 2
USA European American pop 2
```

### Present in Class II only (8 populations)

```
England Northwest Mixed
Hong Kong Chinese HKBMDR. DQ and DP
Ireland Northern
Italy Sardinia pop2
Japan pop 17
USA Colorado Univ Cord Blood Bank Caucasian
USA Colorado Univ Cord Blood Bank Hispanic
Vietnam Kinh  DQB1
```

---

## Summary

| Category | Class I | Class II |
|---|---|---|
| Unique populations (total) | 80 | 86 |
| NMDP-related, present | 20 | 20 |
| NMDP-related, missing | 1 | 1 |
| Non-NMDP, present | 60 | 66 |
