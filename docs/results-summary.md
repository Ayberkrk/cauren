# Results across bridges

One protocol, two bridges so far. Every number here comes from
`cauren_physics/novelty.py` and the study scripts in `tools/`; the case
study pages give the full tables, the data provenance and the limitations.

## The two studies side by side

| | KW51 | Hell Bridge Test Arena |
| --- | --- | --- |
| Structure | Steel bowstring railway bridge, Leuven | Riveted steel truss railway bridge, 35 m, near Trondheim |
| Known change | Strengthening of the diagonal connections (2019) | Eight imposed damage states, bolts removed in deck and bracing connections (2020) |
| Healthy reference | 130 days of hourly frequencies, October to March | One 16-minute shaker recording |
| Healthy holdout | 44 later days, warmer than the reference | One recording five days later |
| Frequencies | Published hourly tracked modes, 5 used | H1 FRF peaks per 60 s window, 5 modes |
| Identification uncertainty, one sample | 0.08 to 0.49% of frequency | 0.13 to 0.51% of frequency |
| Unit scored | Day (median of hourly scores) | 60 s window |
| Alpha | 0.01 (floor 1/131) | 0.1 (floor 1/14) |
| Temperature normalization | Bilinear and Gaussian process | Not possible: one temperature per recording |
| Healthy holdout alarms, best arms | 0 of 44 days (all five arms) | 13 of 13 windows (Mahalanobis, k-NN); PCA 0 of 13 but misses most damage |
| Change detected | Every post-retrofit day, from the first, all arms | Not separable from the healthy holdout |

## What the two bridges show together

1. **The size of the healthy reference decides what can be claimed.**
   KW51's 130 reference days cover months of weather, so a 1% false-alarm
   rate is attainable and held on 44 later days. HBTA's reference is one
   recording; its window scatter (0.13 to 0.51%) understates the variation
   between two healthy sessions (up to 1.7%, 2.6 to 5.7 window sigmas) by a
   factor of about five, and every detector that is sensitive enough to see
   damage also flags the second healthy session. The conformal guarantee
   holds only for data that resembles the reference, and a single session
   does not resemble the next one.

2. **Within-sample scatter is not the uncertainty that matters.** On KW51
   the hour-to-hour identification noise is 15 to 83% of what remains after
   temperature normalization, depending on the mode; the rest is slow
   variation the temperature model does not explain. On HBTA the
   between-session variation dominates. A detector calibrated on the
   short-term scatter alone alarms on healthy data.

3. **A nonparametric detector separates best when the reference is long.**
   On KW51 the k-nearest-neighbour arm, which uses no temperature model and
   no ellipse, separated the retrofit with the widest margin (4.25 times its
   limit on the pre-registered modes, 3.06 on the post-hoc subset) with no
   holdout false alarm. A Gaussian process temperature model widened the
   margin over the bilinear one on the small-shift subset (1.95 against
   1.54) at the cost of a holdout day closer to its limit (0.89).

4. **PCA residuals are fragile in both directions.** On KW51 they alarmed
   on 16% of the warm holdout days once the modes were few; on HBTA they
   stayed quiet on the healthy holdout but also on most damage states.

5. **Calibrate on held-out data.** Scoring the reference with a model fitted
   on it understates how unusual new healthy data looks. On KW51 the
   leave-one-day-out limits are about 8% higher than in-sample ones and
   changed no alarm; on HBTA, with 13 windows, leave-one-out calibration is
   what keeps the limit honest at all.

## What is still missing

- A bridge with real damage **and** a long healthy reference under varying
  temperature. KW51 has the reference but its change is a strengthening;
  HBTA has damage but no reference. Z24 has both and access has been
  requested from KU Leuven.
- Candidates checked while waiting:

  | Data set | Damage | Environment | Access | Fit |
  | --- | --- | --- | --- | --- |
  | Vänersborg bascule bridge, Sweden ([Zenodo 8300495](https://doi.org/10.5281/zenodo.8300495), CC BY 4.0, 550 MB) | Real fracture of a counterweight truss member, March 2023 | Air temperature, wind | Open | Strong on damage, but the records are 64 bridge-opening events rather than ambient vibration, so features would be event signatures rather than natural frequencies |
  | Concrete box-girder bridge ([Zenodo 20522191](https://zenodo.org/records/20522191), CC BY 4.0) | None; a long-term static load test | Temperature in the 9.7 GB file only | Open | Weak: two tracked modes and no damage |
  | Munich two-span test bridge (Jaelani et al., Steel Construction 2023) | Four staged damage scenarios | 18 test days | On request (3 TB) | Strong, not open |
  | Ponte Moesa Campagnola, Switzerland (ASCE, 2025) | Controlled damage progression | Four days | No open repository found | Strong, not open |

- Arms not yet tried: cointegration of the frequencies, and environmental
  models with more covariates than one temperature (humidity, solar
  radiation, traffic).
