# Kandidatuppsats-grundvatten
Detta repo innehåller Python-kod för kandidatuppsatsen om prediktion av grundvattennivåer.

## Struktur
- `kod/visuell_metod/` innehåller kod för WSP:s visuella metod.
- `kod/regressionsmodeller/` innehåller kod för enkel och multipel regressionsmodell.
- `kod/state_space/` innehåller kod för state space-modellen med kalmanfilter.
- `data/` används lokalt för indata.
- `resultat/` används lokalt för figurer och tabeller som genereras av koden.

## Data
Datafilerna ingår inte i repot eftersom de är projektspecifika. 
För att köra koden behöver följande mappar skapas lokalt:

data/observationsror/
data/visuell_metod/

Placera observationsrörens csv-filer i data/observationsror/ och de visuellt matchade filerna i data/visuell_metod/.

## Paket
Installera paket med:

```bash
pip install -r requirements.txt
