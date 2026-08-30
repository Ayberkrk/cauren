# Cauren Civil Architecture

`Raw data -> normalization -> civil router -> cauren-civil -> explainable diagnosis`

Bu surum ASCE JCCE icin tek-sektor civil/construction arastirma hattina indirgenmistir.
Sistem artik bina, insaat, permit, inspection, infrastructure ve risk fusion uzerine kuruludur.

## Ana Prensipler
- Tek public agent: `cauren-civil`
- CBS/building payloadlari dogrudan civil agent'a baglanir.
- Cikti karar-destek amaclidir; otomatik yaptirim veya nihai muhendislik karari uretmez.
- Explainability, structural risk, construction readiness ve inspection evidence uzerinden verilir.

## Civil Feature Cekirdegi
- `structural_risk_score`
- `inspection_finding_score`
- `permit_status_score`
- `construction_progress_pct`
- `infrastructure_connection_score`
- `natural_hazard_score`
- `occupancy_safety_score`
- `ground_stability_score`

## JCCE Konumlandirmasi
Bu mimari, kadastro/CBS, permit ve construction-status verilerini risk-temelli explainable diagnostics ile birlestiren bir civil engineering decision-support arastirma prototipi olarak konumlanir.
