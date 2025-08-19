import functools
import inspect
from enum import Enum
from typing import Callable

from pint import Quantity
from pint.errors import DimensionalityError, PintTypeError


def validate_units(enum_aware=True, ignore_kwargs=None):
    """Validate dimensionality of a function's arguments.
    Dimensionality of set kwargs have same dimensionality as their defaults.

    Note: This is different than Pint's ureg.check decorator as it applied to kwargs.

    Args:
        enum_aware: If True, allow decorator to "reach into" enum to determine
        quantity dimensionality of any.
        ignore_kwargs: Iterables of parameter names to skip validation for.

    Returns:
        Decorated function
    """
    # TODO: Merge with ureg.check as to validate args and kwargs

    ignore_kwargs = ignore_kwargs or []

    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            signature = inspect.signature(func)
            params = signature.parameters.items()

            # Filter out all params that don't correspond to a kwarg that was passed in
            params = filter(lambda it: it[0] in kwargs and it[0] not in ignore_kwargs, params)

            # Filter out all params that don't (or can't) have defaults
            params = filter(lambda it: it[1].kind != inspect.Parameter.VAR_POSITIONAL, params)
            params = filter(lambda it: it[1].default != inspect.Parameter.empty, params)

            for param_name, param_value in params:
                value = kwargs.get(param_name)

                # Handle enum values
                if enum_aware and isinstance(value, Enum):
                    # If the default is also an enum, we're good
                    if isinstance(param_value.default, Enum):
                        continue
                    # If the default is a Quantity, extract the enum's value
                    value = value.value

                # Handle Quantity validation
                if isinstance(param_value.default, Quantity):
                    if not isinstance(value, Quantity):
                        raise PintTypeError(
                            f"Expected a quantity for keyword-argument {param_name} with "
                            f"a dimensionality of {param_value.default.dimensionality} instead "
                            f"got a value of type {type(value)}."
                        )
                    if not value.check(param_value.default.dimensionality):
                        raise DimensionalityError(value, param_value.default)

            return func(*args, **kwargs)

        return inner

    return wrapper
