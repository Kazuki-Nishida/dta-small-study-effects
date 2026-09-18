# Reconstruction of the IPG tables

The IPG source (Goodacre et al., *Health Technol Assess* 2006;10(15), doi:10.3310/hta10150,
open access) reports each cohort's sensitivity and specificity as a proportion rounded to two
decimals with an exact 95% interval, but not the group sizes. The scripts here recover integer
2 x 2 tables by inversion; the method is described in `../README.md` and in Section D.2 of the
supplementary material.

- `hta_text.py` — the enumeration of the admissible integer pairs (count, group size) for a
  printed triple (proportion, lower limit, upper limit), and the parsing of publication years
  from the monograph's reference list.
- `extract_ipg.py` — parses Figures 69 (sensitivity) and 70 (specificity) of the monograph text
  and writes `ipg_min.csv` and `ipg_max.csv` (the two extreme admissible reconstructions of
  each cohort). Inputs (not distributed): `hta1015.txt` and `hta1015_raw.txt`, the
  `pdftotext -layout` and `pdftotext -raw` conversions of the monograph, placed in this directory.
- `derive_ipg_max.py` — rebuilds the maximum-count endpoint from `ipg_min.csv` alone (it
  re-enumerates the candidate sets implied by the minimum-count table, so it needs no source
  text; about one minute): `python derive_ipg_max.py` writes `../ipg_max.csv`.

The data files used by the analyses are the copies in the parent directory (`data/`).
