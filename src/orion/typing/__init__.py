import typing

try:

    if typing.TYPE_CHECKING:
        import torch

except ImportError as err:

    class torch:
        """torch module stub"""
        class nn:
            """toch nn module stub"""
            class Module:
                """torch module stub"""
                pass

        def __getattr__(self) -> None:
            raise err
