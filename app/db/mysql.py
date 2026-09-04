"""MySQL 访问层：SQLAlchemy Core engine，参数化查询，禁止拼接 SQL。"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from app.config import get_settings


def get_engine() -> Engine:
    s = get_settings()
    url = URL.create(
        "mysql+pymysql",
        username=s.mysql_user,
        password=s.mysql_password,
        host=s.mysql_host,
        port=s.mysql_port,
        database=s.mysql_db,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)
