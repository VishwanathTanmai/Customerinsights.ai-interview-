from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import PG_DSN

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(PG_DSN, min_size=1, max_size=10, kwargs={"row_factory": dict_row}, open=True)
    return _pool


@contextmanager
def conn():
    with pool().connection() as c:
        yield c


def connect() -> psycopg.Connection:
    return psycopg.connect(PG_DSN, row_factory=dict_row)


def init_schema() -> None:
    sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with connect() as c:
        c.execute(sql)
        c.commit()
