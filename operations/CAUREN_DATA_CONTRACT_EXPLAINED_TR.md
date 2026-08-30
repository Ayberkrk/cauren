# Cauren Civil Data Contract

Bu repo artik tek-sektor civil/construction odagindadir.

## Agent
- `cauren-civil`
- sektor: `civil`
- amac: building/construction diagnostics and decision support

## Temel Ozellikler
- `structural_risk_score`
- `inspection_finding_score`
- `permit_status_score`
- `construction_progress_pct`
- `infrastructure_connection_score`
- `natural_hazard_score`
- `occupancy_safety_score`
- `ground_stability_score`

## CBS / Building Payload
CBS bina kayitlari normalize edilerek civil sensor penceresine donusturulur.
Kaynak baglaminda su alanlar korunur:
- geometry
- address
- administrative_unit
- project_permit
- construction_status
- infrastructure_connections
- risk_assessments
- cadastral_reference

## Cikti Durusu
Sistem risk ve anomaly family uretir, fakat karar verme otoritesi degildir.
Uretilen sonuclar saha incelemesi, permit review ve risk-onceliklendirme icin decision-support niteligindedir.
