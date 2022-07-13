"""
Non permanent database
======================

Implement non permanent version of :class:`orion.core.io.database.Database`

"""
import copy
from collections import defaultdict

from orion.core.io.database import Database, DatabaseError, DuplicateKeyError
from orion.core.utils.flatten import flatten, unflatten


def _convert_keys_to_name(keys):
    index = []

    if len(keys) == 1 and keys[0] == "_id":
        index = "_id_"
    else:
        index = "_".join(f"{k}_1" for k in keys)

    return index


# pylint: disable=too-many-public-methods
class EphemeralDB(Database):
    """Non permanent database

    This database is meant for debugging purposes. It only lives through one execution and all
    information saved during it is lost when the process is terminated.

    .. seealso:: :class:`orion.core.io.database.Database` for more on attributes.

    """

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}()"

    @property
    def is_connected(self):
        """Return true, always."""
        return True

    def initiate_connection(self):
        """Create the dictionary which serve as an ephemeral database"""
        self._db = defaultdict(EphemeralCollection)

    def close_connection(self):
        """Remove the dictionary"""
        self._db = None

    def ensure_index(self, collection_name, keys, unique=False):
        """Create given indexes if they do not already exist in database.

        Indexes are only created if `unique` is True.
        """
        self._db[collection_name].create_index(keys, unique=unique)

    def index_information(self, collection_name):
        """Return dict of names and sorting order of indexes"""
        return self._db[collection_name].index_information()

    def drop_index(self, collection_name, name):
        """Remove index from the database"""
        self._db[collection_name].drop_index(name)

    def write(self, collection_name, data, query=None):
        """Write new information to a collection. Perform insert or update.

        .. seealso:: :meth:`orion.core.io.database.Database.write` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        if query is None:
            # We can assume that we do not want to update.
            # So we do insert_many instead.
            if type(data) not in (list, tuple):
                data = [data]
            return dbcollection.insert_many(documents=data)

        update_data = {"$set": data}

        return dbcollection.update_many(query=query, update=update_data)

    def read(self, collection_name, query=None, selection=None):
        """Read a collection and return a value according to the query.

        .. seealso:: :meth:`orion.core.io.database.Database.read` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        dbdocs = dbcollection.find(query, selection)

        return dbdocs

    def read_and_write(self, collection_name, query, data, selection=None):
        """Read a collection's document and update the found document.

        Returns the updated document, or None if nothing found.

        .. seealso:: :meth:`orion.core.io.database.Database.read_and_write` for
                     argument documentation.

        """
        dbdoc = self.read(collection_name, query)
        if not dbdoc:
            return None

        id_query = {"_id": dbdoc[0]["_id"]}
        self.write(collection_name, data, id_query)
        return self.read(collection_name, id_query)[0]

    def count(self, collection_name, query=None):
        """Count the number of documents in a collection which match the `query`.

        .. seealso:: :meth:`orion.core.io.database.Database.count` for argument documentation.

        """
        dbcollection = self._db[collection_name]
        return dbcollection.count(query=query)

    def remove(self, collection_name, query):
        """Delete from a collection document[s] which match the `query`.

        .. seealso:: :meth:`orion.core.io.database.Database.remove` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        return dbcollection.delete_many(query=query)

    @classmethod
    def get_defaults(cls):
        """Get database arguments needed to create a database instance.

        .. seealso:: :meth:`orion.core.io.database.Database.get_defaults`
                     for argument documentation.

        """
        return {}


class EphemeralCollection:
    """Non permanent collection

    This collection is meant for debugging purposes within the EphemeralDB.

    .. seealso:: :class:`orion.core.io.database.ephemeraldb.EphemeralDB` for database object.

    """

    def __init__(self):
        """Initialise the collection, with no documents and only _id unique index."""
        self._documents = []
        self._indexes = {}
        self.create_index("_id", unique=True)

    def create_index(self, keys, unique=False):
        """Create given indexes if they do not already exist for this collection.

        Indexes are only created if `unique` is True.
        """
        # turn single key into list for coherence
        if not isinstance(keys, (list, tuple)):
            keys = [(keys, None)]

        keys = tuple(key for (key, order) in keys)
        name = _convert_keys_to_name(keys)
        if unique and name not in self._indexes:
            data = set()

            self._indexes[name] = (keys, data)

            for document in self._documents:
                self._validate_index(document, indexes=[name])
                data.add(tuple(document[key] for key in keys))

    def index_information(self):
        """Return dict of names and sorting order of indexes

        EphemeralCollection may only contain unique indexes.
        """
        return {name: True for name in self._indexes}

    def drop_index(self, name):
        """Remove index from the database

        EphemeralCollection may only contain unique indexes.
        """
        if name not in self._indexes:
            raise DatabaseError(f"index not found with name {name}")

        del self._indexes[name]

    def _register_keys(self, document):
        """Register index values of a new document"""
        for keys, values in self._indexes.values():
            values.add(tuple(document[key] for key in keys))

    def find(self, query=None, selection=None):
        """Find documents in the collection and return a value according to the query.

        .. seealso:: :meth:`orion.core.io.database.Database.read` for argument documentation.

        """
        found_documents = []
        for document in self._documents:
            if document.match(query):
                found_documents.append(document.select(selection))

        return found_documents

    def _validate_index(self, document, indexes=None):
        """Validate index values of a document

        Raises
        ------
        DuplicateKeyError
            If the document contains unique indexes which are already present in the database.

        """
        if indexes is None:
            indexes = self._indexes.keys()

        for name in indexes:
            keys, data = self._indexes[name]
            document_values = tuple(document[key] for key in keys)
            if document_values in data:
                raise DuplicateKeyError(
                    f"Duplicate key error: index={name} value={document_values}"
                )

    def _get_new_id(self):
        """Return max id + 1"""
        if self._documents:
            # NOTE: Custom ids than are not integers should simply not be accounted for
            #       when inferring what next id should be.
            ids = [d["_id"] for d in self._documents if isinstance(d["_id"], int)]
            if not ids:
                return 0

            return max(ids) + 1

        return 1

    def insert_many(self, documents):
        """Add new documents in the collection.

        If the documents do not have a keys `_id`, they are assigned by default
        the max id + 1.

        Raises
        ------
        DuplicateKeyError
            If the document contains unique indexes which are already present in the database.

        """
        for document in documents:
            if "_id" not in document:
                document["_id"] = self._get_new_id()
            ephemeral_document = EphemeralDocument(document)
            self._validate_index(ephemeral_document)
            self._documents.append(ephemeral_document)
            self._register_keys(ephemeral_document)

        return len(documents)

    def update_many(self, query, update):
        """Update documents matching the query

        Raises
        ------
        DuplicateKeyError
            If the update creates a duplication of unique indexes in the database.

        """
        updates = 0
        for document in self._documents:
            if document.match(query):
                document.update(update)
                updates += 1

        return updates

    def _upsert(self, query, update):
        """Insert the document when query was not found.

        If update contains `$set`, then the new document is the combination of query and
        update['$set'], otherwise the new document is `update`.
        """
        if "$set" in update:
            new_document = copy.deepcopy(query)
            new_document.update(update["$set"])
        else:
            new_document = update

        self.insert_many([new_document])

    def count(self, query=None):
        """Count the number of documents in a collection which match the `query`.

        .. seealso:: :meth:`orion.core.io.database.Database.count` for argument documentation.

        """
        return len(self.find(query))

    def delete_many(self, query=None):
        """Delete from a collection document[s] which match the `query`.

        .. seealso:: :meth:`orion.core.io.database.Database.remove` for argument documentation.

        """
        deleted = 0
        retained_documents = []
        for document in self._documents:
            if not document.match(query):
                retained_documents.append(document.to_dict())
            else:
                deleted += 1

        # Reset indexes
        for name, (keys, _) in self._indexes.items():
            self._indexes[name] = (keys, set())

        self._documents = []
        self.insert_many(retained_documents)

        return deleted

    def drop(self):
        """Drop the collection, removing all documents and indexes."""
        self._documents = []
        self._indexes = {}
        self.create_index("_id", unique=True)


class EphemeralDocument:
    """Non permanent document

    This document is meant for debugging purposes within the EphemeralDB.

    .. seealso:: :class:`orion.core.io.database.ephemeraldb.EphemeralDB` for database object.

    """

    operators = {
        "$ne": (lambda a, b: a != b),
        "$in": (lambda a, b: a in b),
        "$gte": (lambda a, b: a is not None and a >= b),
        "$gt": (lambda a, b: a is not None and a > b),
        "$lte": (lambda a, b: a is not None and a <= b),
    }

    def __init__(self, data):
        """Initialise the document with a flattened version of the data"""
        self._data = flatten(data)

    def match(self, query=None):
        """Test if the document corresponds to a given query"""
        if query is None or query == {}:
            return True

        query = flatten(query)
        for key, value in query.items():
            if not self.match_key(key, value):
                return False

        return True

    def _is_operator(self, key):
        return key.split(".")[-1].startswith("$")

    def _get_key_operator(self, key):
        path = key.split(".")
        operator = path[-1]
        key = ".".join(path[:-1])

        if operator not in self.operators:
            raise ValueError(f"Operator '{operator}' is not supported by EphemeralDB")

        return key, self.operators[operator]

    def match_key(self, key, value):
        """Test if a data corresponding to the given key is in agreement with the given
        value based on the operator defined within the key.

        Default operator is equal when no operator is defined.
        Other operators could be $ne, $in, $gte, $gt or $lte. They are defined
        in the last section of the key. For example: `abc.def.$in` or `abc.def.$gte`.
        """
        if self._is_operator(key):
            key, operator = self._get_key_operator(key)

            return key in self and operator(self[key], value)

        return key in self and self[key] == value

    def _validate_keys(self, keys):
        """Verify that all keys are 0 or 1 (with exception of _id) and convert them.

        For simplicity, when keys are 0, the inverse set of keys for 1s is computed.

        .. note ::

            _id is set to 1 if not specified. Only _id may be set to 0 if other keys are set to 1.
        """
        if len(keys) == 1 and keys.get("_id", 0) == 1:
            return keys

        keys_without_id = [key for key in keys if key != "_id"]
        n_keys = sum(keys[key] for key in keys_without_id)
        if n_keys != 0 and n_keys != len(keys_without_id):
            raise ValueError(
                f"Cannot mix selection with 1 and 0s except for _id: {keys}"
            )

        # All given keys are 0 (with possible exception of _id)
        if n_keys == 0:
            new_keys = {key: 1 for key in self._data.keys() if key not in keys}
            new_keys["_id"] = keys.get("_id", 1)
            keys = new_keys

        keys.setdefault("_id", 1)

        return keys

    def select(self, keys):
        """Only select or only drop the specified keys

        For a pair (key, value) in the dictionary, value=0 means the key will not be included
        while value=1 means it will.

        All specified keys should be 0 or 1. They cannot have different values with the exception
        of _id which can be specified to 0 while the others are at 1. The _id field is always
        returned unless specified with 0.

        Parameters
        ----------
        keys: dict
            Pairs of keys and 0 or 1s. When a key is associated with 1, it is kept in the selection,
            otherwise it is dropped.

        """
        if not keys:
            return unflatten(self._data)

        keys = flatten(keys)
        keys = self._validate_keys(keys)

        selection = {}

        def key_is_match(key, selected_key):
            """Test if key matches the selected key

            key_is_match(abc.def.ghi, abc.def.ghi) -> True
            key_is_match(abc.def.ghi, abc.def) -> True
            key_is_match(abc.def.ghi, abc.de) -> False
            key_is_match(abc.def.ghi, xyz) -> False
            """
            return key == selected_key or (
                key.startswith(selected_key) and key.replace(selected_key, "")[0] == "."
            )

        for selected_key, include in filter(lambda item: item[1], keys.items()):
            match = False
            for key, value in self._data.items():
                if include and key_is_match(key, selected_key):
                    match = True
                    selection[key] = value

            if not match:
                selection[selected_key] = None

        return unflatten(selection)

    def update(self, data):
        """Update the values of the document.

        Parameters
        ----------
        data: dict
            Dictionary of data to update the document. If `$set` is in
            the data, the corresponding `data[$set]` will be used instead.

        """
        if "$set" in data:
            unflattened_data = unflatten(self._data)
            for key, value in data["$set"].items():
                if isinstance(value, dict):
                    value = flatten(value)
                unflattened_data[key] = value
            self._data = flatten(unflattened_data)
        else:
            self._data.update(flatten(data))

    def to_dict(self):
        """Convert the ephemeral document to a python dictionary"""
        return self.select({})

    def __getitem__(self, key):
        """Get the item corresponding to the given key in the document"""
        return self._data.get(key, None)

    def __contains__(self, key):
        """Test whether the given key is present in the document"""
        return key in self._data
