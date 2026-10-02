# Study-level data of the two re-analysed reviews

Each file holds one 2 x 2 table per primary study (`TP`, `FN`, `FP`, `TN`, with `n1 = TP + FN`
and `n0 = FP + TN`). Neither dataset contains individual patient data. In the two IPG files the
last column, `flag`, is the extraction-note column of `reconstruction/extract_ipg.py`; it is empty
for every cohort (no manual intervention was needed).

| File | Review | k | Source of the study-level numbers |
|---|---|---|---|
| `fit_crc_refpos.csv` | Faecal immunochemical tests (FIT) for colorectal cancer in average-risk screening populations, studies with the "reference standard: positive" design (Review 1; Grobbee et al., *Cochrane Database Syst Rev* 2022, CD009276.pub2, analysis "Reference standard: positive – FIT – CRC") | 23 | The published counts of the review's own analysis, taken from the review's RevMan data file (Cochrane Library, "Download statistical data", `CD009276StatsDataOnly.rm5`, test TST-037). Exact; no reconstruction. |
| `fit_crc_refpos_characteristics.csv` | the same 23 entries | 23 | Country, period, FIT brand, positivity threshold and follow-up of test negatives, transcribed from the review's "Characteristics of included studies". |
| `ipg_min.csv` | Impedance plethysmography (IPG) for symptomatic deep-vein thrombosis (Review 2; Goodacre et al., *Health Technol Assess* 2006;10(15)) | 42 | Goodacre et al. 2006, Figures 69 and 70 (per-cohort sensitivity and specificity with exact 95% intervals), inverted to integer tables; minimum-count member of each cohort's admissible set. Used in the main text. |
| `ipg_max.csv` | same review | 42 | Maximum-count member of each cohort's admissible set (endpoint bracketing, supplement Section D.2; `reconstruction/derive_ipg_max.py`). |

## Review 1: FIT for colorectal cancer, "reference standard: positive" studies

In these studies participants with a positive FIT were referred for colonoscopy and participants
with a negative FIT were followed, through cancer registries or clinical follow-up, for the
development of an interval cancer (the review's definition; follow-up of one to more than three
years, see the characteristics file). A row is one entry of the review's analysis: 22 studies,
one of which (Chiang 2014) contributes two entries for two FIT brands used in different parts of
the Taiwanese programme (747,076 and 208,929 participants). Denters 2012a and Levi 2011b are the
FIT arms of studies that randomized participants between a guaiac test and a FIT. Four entries
(Kapidzic 2017, Levi 2011b, Parra-Blanco 2010, Robinson 1996) have no false negatives. Study
labels are the review's study IDs (the year suffixes a/b are the review's). The positivity
thresholds are given as the review states them, with its conversions to ug Hb/g in parentheses.

The Cochrane Library's terms for downloaded data (cochranelibrary.com/about/data-download) grant a
non-commercial licence to extract, copy and share the data with attribution; the review's own
summary of these studies and the primary reports are the authoritative sources.

## Review 2: IPG for deep-vein thrombosis, reconstruction of the tables (`reconstruction/`)

The HTA monograph reports each cohort's sensitivity and specificity as a proportion rounded
to two decimals with an exact (Clopper-Pearson) 95% interval, but not the group sizes.
`reconstruction/extract_ipg.py` parses those figures from the text of the monograph and, for
every printed `(p, lower, upper)`, enumerates all integer pairs `(x, n)` with `n <= 4000` whose
proportion and interval both reproduce the printed values (every `x` with `|x/n - p| <= 0.005 + 1e-9`,
robust to half-up versus banker's rounding; the exact Clopper-Pearson limits matched to the printed
values with the same tolerance). The integer tables compatible with a printed triple are not
unique: for the high-accuracy cohorts the admissible group sizes extend to several hundred (a
proportion of 0.97 with interval 0.95 to 0.99 is reproduced by 331 pairs with group sizes from 234
to 517; the largest admissible group size over the 84 coordinates is 606). As an analysis choice,
the candidate pairs of each coordinate are ordered by group size, then by count, and the first
twelve are kept (`hta_text.KEEP`); this truncation is binding for 4 sensitivity and 11 specificity
coordinates. The sensitivity and specificity candidates are then paired under a prevalence between
0.02 and 0.95 (all pairs if none qualifies), and the pairs with the smallest and the largest total
group size (ties by the sensitivity group size) are the two reconstructions carried through the
whole analysis, `ipg_min.csv` and `ipg_max.csv`; 30 of the 42 cohorts admit more than one pair.
`derive_ipg_max.py` rebuilds the maximum-count endpoint by the same rule from `ipg_min.csv` (it
re-enumerates the candidate sets from the printed values implied by the minimum-count table, so it
needs no source text); the rule is shared by the two scripts through `hta_text.pair_candidates` and
`hta_text.endpoints`. The two reconstructions are the extremes within the truncated candidate sets,
not over every table compatible with the printed values. Publication years (`year_pub`) were taken
from the source review's reference list.

To re-run the extraction, obtain the monograph (open access, doi:10.3310/hta10150), convert it
to text with `pdftotext -layout hta1015.pdf hta1015.txt` and `pdftotext -raw hta1015.pdf
hta1015_raw.txt` inside `reconstruction/`, and run `extract_ipg.py`. The text files are not
distributed with this repository.
