IMPORT_ERROR = None
try:
    from orion.storage.sql_impl import SQLAlchemy as SQLAlchemyImpl

    HAS_SQLALCHEMY = True
except ModuleNotFoundError as err:
    IMPORT_ERROR = err
    HAS_SQLALCHEMY = False


if IMPORT_ERROR is not None:
    from orion.storage.base import BaseStorageProtocol

    class SQLAlchemy(BaseStorageProtocol):
        def __init__(self, uri, token=None, **kwargs):
            raise IMPORT_ERROR

else:
    SQLAlchemy = SQLAlchemyImpl
