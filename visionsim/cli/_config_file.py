from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable, Sequence
from typing import Any

import tyro

from visionsim.utils.config_file import ConfigIO


def bind_config_file(fn: Callable[..., Any], argv: Sequence[str]) -> Callable[..., Any]:
    """Bind ``--<param>.load`` and ``--<param>.dump`` handling to a command invocation.

    A function rather than a decorator: the decision depends on the current ``argv``
    and is made in ``main()`` for the selected subcommand only. Commands without a
    config parameter carrying ``load``/``dump`` options, and invocations that use
    neither option, are returned unchanged.

    Args:
        fn (Callable[..., Any]): Command function to wrap.
        argv (Sequence[str]): Command-line arguments, excluding the subcommand name.

    Returns:
        Callable[..., Any]: ``fn``, optionally wrapped with a loaded config default and
        a dump step that runs after tyro parses the arguments.
    """
    hints = typing.get_type_hints(fn, include_extras=True)
    params: list[tuple[str, type[ConfigIO]]] = []
    for name in inspect.signature(fn).parameters:
        typ = hints.get(name)
        if isinstance(typ, type) and issubclass(typ, ConfigIO):
            params.append((name, typ))
    if not params:
        return fn
    # Help must reach tyro without reading a config file first.
    if any(token in ("-h", "--help") for token in argv):
        return fn
    # Avoid a second parse when neither option is present; plain invocations stay untouched.
    options = [f"--{name.replace('_', '-')}.{field}" for name, _ in params for field in ("load", "dump")]
    if not any(token == option or token.startswith(f"{option}=") for token in argv for option in options):
        return fn

    configs = _parse_config_values(fn, params, argv)
    loaded: dict[str, ConfigIO] = {}
    for name, typ in params:
        if (path := configs[name].load) is not None:
            loaded[name] = typ.from_file(path)
    needs_dump = any(config.dump for config in configs.values())

    if not loaded and not needs_dump:
        return fn
    if needs_dump:
        fn = _with_dump(fn, [name for name, _ in params])
    if loaded:
        fn = functools.partial(fn, **loaded)
    return fn


def _parse_config_values(
    fn: Callable[..., Any], params: list[tuple[str, type[ConfigIO]]], argv: Sequence[str]
) -> dict[str, ConfigIO]:
    # Parse once to discover load/dump paths; the command is parsed again with the
    # loaded defaults. The probe binds the configs so required arguments stay required.
    signature = inspect.signature(fn)
    parsed: dict[str, Any] = {}

    @functools.wraps(fn)
    def probe(*args: Any, **kwargs: Any) -> None:
        parsed.update(signature.bind(*args, **kwargs).arguments)

    tyro.cli(functools.partial(probe, **{name: typ() for name, typ in params}), args=argv)
    return {name: parsed[name] for name, _ in params}


def _with_dump(fn: Callable[..., Any], names: Sequence[str]) -> Callable[..., Any]:
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name in names:
            config = bound.arguments.get(name)
            if isinstance(config, ConfigIO) and config.dump is not None:
                config.to_file(config.dump)
        return fn(*args, **kwargs)

    return wrapper
