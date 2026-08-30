# Civil LLM Data Layout

Bu klasor yalnizca `cauren-civil` icin aciklanabilir decision-support egitim kayitlarini tutar.

## Dizinler

- `data/raw/notebooklm_exports/`: kaynak notlar veya saha anlatilari
- `data/templates/`: civil inceleme ve raporlama sablonlari
- `data/processed/`: dogrulanmis JSONL egitim kayitlari

## Kayit Temasi

Her kayit su baglamlardan birini hedeflemelidir:

- structural risk review
- permit/compliance review
- construction progress review
- inspection finding synthesis
- hazard and ground stability review

## Kalite Kurallari

- `metadata.agent_id == "cauren-civil"` olmali.
- Cikti karar-destek tonunda kalmali; nihai muhendislik karari iddia etmemeli.
- Kayitlar civil sahaya ait acik evidence veya kamu verisi baglami tasimali.
