from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import tyro
import yaml
from typing_extensions import Self

_RESERVED_FIELDS = ("load", "dump")
_UNIONS = (typing.Union, types.UnionType)


@dataclass
class ConfigIO:
    """Mixin adding ``load`` and ``dump`` options to a CLI-facing config dataclass.

    Inheriting from this class makes a config dataclass loadable from a YAML or JSON
    file through its CLI, e.g. ``--config.load path/to/file.yaml``, and lets the
    effective configuration be written back with ``--config.dump``.
    """

    load: Annotated[
        Path | None,
        tyro.conf.arg(help="Load options from this YAML or JSON file before parsing the remaining flags."),
    ] = None
    dump: Annotated[
        Path | None,
        tyro.conf.arg(help="Write the effective options to this YAML file before running."),
    ] = None

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load a configuration from a YAML or JSON file.

        The file mirrors the dataclass field names and nesting. Unknown keys are rejected
        with the offending key path; values are parsed by tyro itself, so the accepted
        syntax is exactly the one the command line accepts.

        Args:
            path (Path): File to read. YAML is used for parsing, which also accepts JSON.

        Returns:
            A configuration built from the file, with missing fields taking their
            dataclass defaults.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file cannot be parsed or contains unknown keys.
            TypeError: If the file does not contain a mapping at the top level.
            SystemExit: If tyro rejects a value, after printing its usual error message.
        """
        source = Path(path)
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError(f"Could not parse config file {source}: {e}") from e

        if not isinstance(data, dict):
            raise TypeError(f"Config file {source} must contain a mapping at the top level, got {type(data).__name__}.")
        return tyro.cli(cls, default=cls(), args=_to_args(cls, data, str(source)))

    def to_file(self, path: Path) -> None:
        """Write this configuration to a YAML file.

        Field declaration order is preserved, ``Path`` values are written as strings and
        tuple values as lists. The ``load`` and ``dump`` options themselves are omitted so
        the result can be passed back to :meth:`from_file`.

        Args:
            path (Path): Destination file; parent directories are created.
        """
        data = dataclasses.asdict(typing.cast(Any, self))
        data.pop("load", None)
        data.pop("dump", None)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.dump(data, Dumper=_Dumper, sort_keys=False), encoding="utf-8")


class _Dumper(yaml.SafeDumper):
    """SafeDumper that serializes ``Path`` values as strings."""


_Dumper.add_multi_representer(Path, lambda dumper, path: dumper.represent_str(str(path)))


def _to_args(cls: type[Any], data: dict[str, Any], where: str, prefix: str = "") -> list[str]:
    """Convert a config mapping into tyro option tokens for ``cls``."""
    hints = typing.get_type_hints(cls, include_extras=True)
    data = {key.replace("-", "_"): value for key, value in data.items()}
    fields = {f.name for f in dataclasses.fields(typing.cast(Any, cls))}

    if unknown := sorted(set(data) - fields):
        raise ValueError(f"{where}: unknown option(s) {unknown}; valid options are {sorted(fields)}")

    if issubclass(cls, ConfigIO):
        reserved = sorted(set(data) & set(_RESERVED_FIELDS))
        if reserved:
            raise ValueError(f"{where}: {reserved} are command-line options and cannot be set in a config file")

    args: list[str] = []
    for key, value in data.items():
        if value is None:
            continue
        name = f"{prefix}{key}"
        where_key = f"{where}:{key}"
        typ, optional = _strip(hints[key])
        if isinstance(value, dict):
            if not (isinstance(typ, type) and dataclasses.is_dataclass(typ)):
                raise ValueError(f"{where_key}: expected a mapping, got {type(value).__name__}")
            args.extend(_to_args(typ, value, where_key, prefix=f"{name}."))
        elif isinstance(value, bool) and not optional:
            args.append(f"--{prefix}{'' if value else 'no-'}{key}")
        elif isinstance(value, bool):
            args.append(f"--{name}={value}")
        elif isinstance(value, (list, tuple)):
            items = [str(item) for item in value]
            args.extend([f"--{name}={items[0]}", *items[1:]] if items else [f"--{name}"])
        else:
            args.append(f"--{name}={value}")
    return args


def _strip(typ: Any) -> tuple[Any, bool]:
    """Unwrap ``Annotated``/``Optional`` and report whether ``None`` was allowed."""
    while typing.get_origin(typ) is typing.Annotated:
        typ = typing.get_args(typ)[0]
    origin = typing.get_origin(typ)
    if origin in _UNIONS:
        options = [opt for opt in typing.get_args(typ) if opt is not type(None)]
        if len(options) == 1:
            return options[0], len(options) != len(typing.get_args(typ))
    return typ, False
