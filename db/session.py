from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

DB_PATH = "data/dfs.sqlite"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,  # turn on for debugging
    future=True,
)


# VERY IMPORTANT: enable foreign keys in SQLite
@event.listens_for(engine, "connect")
def enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)
