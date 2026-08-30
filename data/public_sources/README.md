# Civil Public Sources

Bu klasor, `cauren-civil` egitim/test hattini besleyen resmi kamu verisi kaynaklarini toplar.

## Politika

- Yalnizca ucretsiz, arastirmada atifla kullanilabilir ve yeniden indirilebilir kaynaklar listelenir.
- Ana truth kaynaklari resmi kamu kurumu verileridir.
- OSM ve bina footprint benzeri yardimci kaynaklar truth yerine enrichment olarak kullanilir.

## Akis

1. `source_manifest.json` icindeki kaynaklar ve lisans notlari incelenir.
2. `tools/download_civil_public_data.py` ile indirilebilir artefaktlar cekilir.
3. Ham veriler normalize edilerek `normalized/civil_public_core_assets.csv` uretilir.
4. `tools/build_cauren_civil_dataset.py` ile kanonik egitim veri seti olusturulur.

## Beklenen Normalize CSV Kolonlari

- `asset_id`
- `site_id`
- `split`
- `seq_len`
- `sampling_hz`
- `structural_risk_score`
- `inspection_finding_score`
- `permit_status_score`
- `construction_progress_pct`
- `infrastructure_connection_score`
- `natural_hazard_score`
- `occupancy_safety_score`
- `ground_stability_score`

Opsiyonel kolonlar:

- `building_height_m`
- `footprint_area_m2`
- `soil_settlement_mm`
- `utility_disruption_score`
