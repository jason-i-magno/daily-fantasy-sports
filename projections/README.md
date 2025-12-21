# NBA Projections Ingestion (Local-Only)

Python tooling to ingest NBA projection data (ETR, RotoGrinders) into a local SQLite database for personal analysis. No credentials are stored, no login automation, no web scraping.

## Project layout
```
projections/
  config.py          # app/provider config defaults
  ingest.py          # CLI entrypoint
  providers/
    base.py          # ProjectionProvider interface
    etr.py           # ETR implementation (CSV or cookie-backed JSON)
    rg.py            # RG implementation (CSV or cookie-backed JSON)
  db/
    models.py        # SQLAlchemy models
    session.py       # engine/session factory
    migrate.py       # create tables
```

## Setup
- Python 3.11+
- Dependencies: `requests`, `pandas`, `sqlalchemy`

Install (editable):
```bash
pip install -r requirements.txt
```

Initialize the database (SQLite by default at `sqlite:///projections.db`):
```bash
python projections/db/migrate.py
```

## Adding a provider
1. Implement `ProjectionProvider` in `projections/providers/<name>.py` with:
   - `fetch_raw(self, slate_id) -> pd.DataFrame`
   - `normalize(self, df, slate_id) -> pd.DataFrame` producing the common schema.
2. Use the shared normalized fields:
   - slate_id (TEXT), player_name, team, position, projection (FLOAT), ceiling (FLOAT nullable), floor (FLOAT nullable), minutes (FLOAT nullable), ownership (FLOAT nullable), source (TEXT), pulled_at (TIMESTAMP UTC).
3. Do not add optimizer-specific assumptions. Keep it defensive and testable.

## Running an ingest
Examples:
```bash
# ETR from CSV
python projections/ingest.py --slate-id 2025-01-15-NBA --provider etr --etr-csv data/raw/etr/nba_main_dk_etr_projections.csv

# RG from CSV
python projections/ingest.py --slate-id 2025-01-15-NBA --provider rg --rg-csv data/raw/rg/nba_main_dk_rg_projections.csv

# Both providers
python projections/ingest.py --slate-id 2025-01-15-NBA --provider all --etr-csv ... --rg-csv ...
```

For authenticated JSON endpoints, supply the URL and a user-provided session cookie:
```bash
python projections/ingest.py --slate-id 2025-01-15-NBA --provider etr --etr-url https://... --etr-cookie "session=abc;"
```

## Querying the DB
Using SQLite CLI:
```bash
sqlite3 projections.db "SELECT source, COUNT(*) FROM player_projections GROUP BY source;"
```

With SQLAlchemy:
```python
from projections.db.session import get_session_maker
from projections.db.models import PlayerProjection

Session = get_session_maker()
with Session() as s:
    rows = s.query(PlayerProjection).filter_by(slate_id="2025-01-15-NBA").all()
```

## Idempotency
- Natural uniqueness on `(slate_id, player_name, source)`.
- Ingest uses upsert to avoid duplicates on repeated runs for the same slate+source.

## Non-goals
- No optimizer, no lineup generation, no projection blending.
- No cloud deployment or scheduling.
- No login automation or HTML scraping.
