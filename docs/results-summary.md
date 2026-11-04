# Results across bridges

One protocol, three bridges. Every number here comes from
`cauren_physics/novelty.py` and the study scripts in `tools/`; the case
study pages give the full tables, the data provenance and the limitations.

## The three studies side by side

| | KW51 | Hell Bridge Test Arena | Vänersborg |
| --- | --- | --- | --- |
| Structure | Steel bowstring railway bridge, Leuven | Riveted steel truss railway bridge, 35 m, near Trondheim | Riveted steel bascule railway bridge, south-west Sweden |
| Known change | Strengthening of the diagonal connections (2019) | Eight imposed damage states: bolts removed in deck and bracing connections (2020) | Real fracture of a counterweight truss member (March 2023) |
| Healthy reference | 130 days, October to March | One 16-minute shaker recording | 40 openings over 11 days |
| Healthy holdout | 44 later days, warmer than the reference | One recording five days later | 17 later openings, colder nights than the reference |
| Features | Hourly tracked frequencies, 5 modes | H1 FRF peaks per 60 s window, 5 modes | Strain change from closed at two leaf angles, 15 gauges |
| Sample uncertainty | 0.08 to 0.49% of frequency per hour | 0.13 to 0.51% of frequency per window | Holdout shifts within 0.6 reference sigmas |
| Unit scored, alpha | Day, 0.01 | 60 s window, 0.1 | Opening, 0.05 |
| Healthy holdout alarms | 0 of 44 days, every arm | 13 of 13 windows (Mahalanobis, k-NN) | 0 of 17 (Mahalanobis), 5 of 17 (PCA) |
| Change detected | Every post-retrofit day, every arm | Not separable from the healthy holdout | Pre-registered: no. Post hoc, one erratic gauge excluded: 6 of 6 (Mahalanobis), 1 healthy false alarm |

## What the three bridges show together

1. **The healthy reference must span the variation the detector will
   meet.** KW51's 130 reference days cover a winter, so a 1% false-alarm
   rate is held on 44 later days. HBTA's reference is one recording: its
   window scatter understates the variation between two healthy sessions
   (up to 1.7%, 2.6 to 5.7 window sigmas) about fivefold, and every
   detector sensitive enough to see damage flags the next healthy session.
   On Vänersborg the PCA arm alarms on 29% of healthy openings, which
   include nights colder than any reference opening. The conformal
   guarantee holds only for data that resembles the reference.

2. **The features can see damage that the detectors miss.** At Vänersborg
   the two gauges next to the crack moved by 5.4 and 4.0 reference sigmas,
   and SG10 left the reference range in every opening after the fracture.
   The pre-registered detectors still missed it, because one erratic gauge
   inflated the calibration of a 30-feature distance. Data quality of the
   reference, checked on the reference alone, decided the outcome.

3. **On a long reference, the arms that model the healthy structure's
   shape detect the smallest changes.** Injecting known frequency drops
   into KW51's healthy holdout, cointegration and k nearest neighbours
   detect a 0.5% drop of the best-observed modes, and a 0.5% uniform drop
   of all modes, on at least 90% of days; the Mahalanobis arms need 1%,
   with or without temperature normalization. Modes with the largest
   identification noise hide drops of up to 2% from almost every arm.

4. **The retrofit alone ranks the arms differently from known drops.**
   Daily averaging of the residuals looked best on the small-shift retrofit
   subset (2.07 times its limit against 1.54) but is the least sensitive to
   injected drops (2%). Cointegration has the widest retrofit margin but
   alarms on 16% of warm holdout days with three modes. One change of one
   kind is not enough to choose a method.

5. **More covariates and sequential tests do not fix a reference that is
   too short.** Weather-station air temperature, humidity, radiation and
   wind changed KW51's margins by less than 0.1. A CUSUM of daily evidence,
   calibrated for one false alarm a year on days like the reference, raised
   a false alarm 19 to 23 days into the healthy holdout for six of seven
   arms: it accumulates the small, persistent difference between a warm
   spring and the winter reference.

6. **Calibrate on held-out data.** Scoring reference samples with a model
   fitted on them understates how unusual new healthy data looks. On KW51
   leave-one-day-out limits are about 8% higher and changed no alarm; with
   13 HBTA windows or 40 Vänersborg openings it is what keeps the limit
   honest.

## What is still missing

- A bridge with real damage **and** a long healthy reference spanning the
  seasons. Vänersborg comes closest but its reference covers 11 winter
  days. Z24 has a year of monitoring before progressive damage tests;
  access has been requested from KU Leuven.
- Other candidates checked:

  | Data set | Damage | Environment | Access | Fit |
  | --- | --- | --- | --- | --- |
  | Concrete box-girder bridge ([Zenodo 20522191](https://zenodo.org/records/20522191), CC BY 4.0) | None; a long-term static load test | Temperature in the 9.7 GB file only | Open | Weak: two tracked modes and no damage |
  | Munich two-span test bridge (Jaelani et al., Steel Construction 2023) | Four staged damage scenarios | 18 test days | On request (3 TB) | Strong, not open |
  | Ponte Moesa Campagnola, Switzerland (ASCE, 2025) | Controlled damage progression | Four days | No open repository found | Strong, not open |

- A rule for choosing among arms before seeing the change. The synthetic
  drops on KW51 are the fairest comparison so far, but they come from one
  bridge and one season.
