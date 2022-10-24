"""Example usage and tests for :mod:`orion.core.utils.module_import`."""
import pytest

from orion.core.utils.module_import import ImportOptional


class TestImportOptional:
    def test_package_name_default_extra(self):
        with ImportOptional("PacK") as import_optional:
            pass

        assert import_optional.package == "PacK"
        assert import_optional.extra_dependency == "pack"

    def test_package_name_custom_extra(self):
        with ImportOptional("PacK", "age") as import_optional:
            pass

        assert import_optional.package == "PacK"
        assert import_optional.extra_dependency == "age"

    def test_import_error_caugth(self):
        with pytest.raises(RuntimeError, match="abc"):
            with ImportOptional("PacK"):
                raise RuntimeError("abc")

        with ImportOptional("PacK"):
            raise ImportError

        with ImportOptional("PacK"):
            raise ModuleNotFoundError()

    def test_failed(self):
        with ImportOptional("PacK") as import_optional:
            pass

        assert not import_optional.failed

        with ImportOptional("PacK") as import_optional:
            raise ImportError

        assert import_optional.failed

    def test_ensure(self):
        with ImportOptional("PacK") as import_optional:
            pass

        import_optional.ensure()

        with ImportOptional("PacK", "age") as import_optional:
            raise ImportError

        error_message = (
            "The package `PacK` is not installed. "
            r"Install it with `pip install orion\[age\]`."
        )
        with pytest.raises(ImportError, match=error_message):
            import_optional.ensure()
