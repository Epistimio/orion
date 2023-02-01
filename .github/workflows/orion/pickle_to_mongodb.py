from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import setup_storage


def main():
    storage = setup_storage(
        {
            "database": {
                "host": ".github/workflows/orion/db_dashboard_full_with_uncompleted_experiments.pkl",
                "type": "pickleddb",
            }
        }
    )
    pickle_db = storage._db
    assert isinstance(pickle_db, PickledDB)
    mongo_db = MongoDB(name="orion_dashboard_test")
    with pickle_db.locked_database(write=False) as database:
        for collection_name in database._db.keys():
            print(f"[{collection_name}]")
            # Clean Mongo DB first
            for element in mongo_db.read(collection_name):
                mongo_db.remove(collection_name, {"_id": element["_id"]})
            # Fill Mongo DB
            data = database.read(collection_name)
            mongo_db.write(collection_name, data)
    print("Pickle to Mongo DB done.")


if __name__ == "__main__":
    main()
