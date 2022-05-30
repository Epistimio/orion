import inspect
from dataclasses import asdict, fields, is_dataclass
from typing import (
    Any,
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
)
from collections.abc import ItemsView
from logging import getLogger as get_logger
from orion.core.utils.flatten import flatten, unflatten

logger = get_logger(__name__)


class SerializableMixin(Mapping[str, Any]):
    """Mixin that makes dataclasses behave like dicts and easy to serialize."""

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        """NOTE: This will not work for frozen dataclasses."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Returns the item / attribute with the given name if present, else `default`."""
        return getattr(self, key, default)

    def __iter__(self) -> Iterator[str]:
        return iter([f.name for f in fields(self)])

    def __len__(self) -> int:
        return len(fields(self))

    def keys(self) -> KeysView[str]:
        return KeysView({f.name: getattr(self, f.name) for f in fields(self)})

    def items(self) -> ItemsView[str, Any]:
        """Returns an iterator over the items/attributes in this object."""
        return ItemsView({f.name: getattr(self, f.name) for f in fields(self)})

    def asdict(self) -> Dict[str, Any]:
        """Returns a dict version of this dataclass."""
        return asdict(self)

    def replace(self, **kwargs):
        """Replaces one or more values in this dataclass.

        Can also replace a nested entry. For example:

        >>> from dataclasses import dataclass
        >>> from typing import List
        >>> @dataclass
        ... class Person(SerializableMixin):
        ...     name: str
        ...     age: int = 18
        ...
        >>> @dataclass
        ... class Party(SerializableMixin):
        ...    host: Person
        ...    guests: List[Person]
        ...
        >>> party = Party(host=Person(name="John"), guests=[])
        >>> party
        Party(host=Person(name='John', age=18), guests=[])
        >>> party.replace(**{"host.age": 20})
        Party(host=Person(name='John', age=20), guests=[])
        """
        flattened_dict = self.flatten()
        flattened_changes = flatten(kwargs)
        flattened_dict.update(flattened_changes)
        unflattened_dict = unflatten(flattened_dict)
        return self.from_dict(unflattened_dict)

    def flatten(self, sep: str = ".") -> dict:
        """Returns a flattened, dictionary version of this dataclass, using `sep` as a separator.

        Returns
        -------
        Dict
            A dictionary version of this config, with `sep` as a separator.
        """
        return flatten(self.asdict(), sep=sep)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        """Create an instance of this class from the given dictionary."""
        config_dict = config_dict.copy()

        constructor_arguments = {}
        for field in fields(cls):
            if field.name not in config_dict:
                # The field isn't in the dict, so let the dataclass constructor use the default or
                # call the default factory.
                continue

            field_type = field.type
            if isinstance(field_type, str):
                # Try to resolve the forward reference to a dataclass that is a subclass of
                # SerializableMixin.
                # Find all the classes that have a name that matches the given field type.
                from typing import get_type_hints

                try:
                    type_hints = get_type_hints(cls)
                    field_type = type_hints[field.name]
                except (ValueError, NameError) as err:
                    logger.debug(
                        f"Unable to resolve the type hint string {field_type} for field "
                        f"{field.name}: {err}\n Values will remain dictionaries."
                    )

            if (
                is_dataclass(field_type)
                and inspect.isclass(field_type)
                and issubclass(field_type, SerializableMixin)
            ):
                # The field is itself a dataclass type, so we recursively call from_dict
                # passing in the corresponding value.
                field_config_dict: Dict[str, Any] = config_dict.pop(field.name)
                field_value = field_type.from_dict(field_config_dict)
            else:
                field_value = config_dict.pop(field.name)

            # TODO: Could also perhaps extend to tuples of dataclasses.
            if (
                isinstance(field_value, (list, tuple))
                and field_value
                and isinstance(field_value[0], dict)
            ):
                raise NotImplementedError(
                    "Only currently support fields that are containers of primitives No dicts or "
                    "nested collections of configs."
                )

            constructor_arguments[field.name] = field_value

        # Add the leftover values to the dict of constructor arguments, which will produce a
        # clear error if there are extras (e.g. "Type <...> got an unexpected keyword argument
        # 'foo'").
        if config_dict:
            for key, value in config_dict.items():
                assert key not in constructor_arguments
                constructor_arguments[key] = value
        return cls(**constructor_arguments)  # type: ignore
