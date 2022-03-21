try:
    import torch

    HAS_TORCH = True
    TORCH_ERROR = None

except ImportError as err:
    TORCH_ERROR = err
    HAS_TORCH = False

    class torch:
        """torch module stub"""

        class nn:
            """toch nn module stub"""

            class Module:
                """torch module stub"""

                pass

        def __getattr__(self) -> None:
            raise TORCH_ERROR
