import functools
import inspect
import re
import sys
from typing import Callable

from invoke import Context, Parser, ParserContext, Program, Task
from tqdm.auto import tqdm


def ignore_context(func: Callable) -> Callable:
    """Decorator that modifies a function's signature to include a `context` variable as first argument.
    
    Args:
        func: Function to decorate, should be a callable

    :returns:
        Decorated function with context argument, which is ultimately ignored when calling the original function.
    """

    # Based on https://github.com/andrewtbiehl/gaslines/pull/82/files
    # See: https://github.com/pyinvoke/invoke/issues/445
    #RY. Create paramter object named context with type positional only
    context_param = inspect.Parameter("context", inspect.Parameter.POSITIONAL_ONLY)
    #RY. create signature object with signature of func callable
    signature = inspect.signature(func)
    #RY. Stores iterable of Parameter objects 
    params = signature.parameters.values()
    ctx_params = (context_param, *params)
    #RY. replace paramters of original signature
    func.__signature__ = signature.replace(parameters=ctx_params)

    #RY. decorator to wrapper function that copies attributes like name and docstrings
    @functools.wraps(func)
    #RY. ignore context
    def inner(context, *args, **kwargs):
        del context  
        func(*args, **kwargs)

    return inner


def pass_if_present(func: Callable) -> Callable:
    """Decorator that passes keyword arguments to a funciton  only if they were explicitly set on CLI.

    Args:
        func: Function to be decorated, should be a callable

    :returns:
        Decorated function with kwargs arguments modified and filtered.
    """
    # Pass kwargs to func only if they were explicitly set on CLI
    # This is useful especially for boolean variables, i.e: if "kw" not in
    # kwargs then variable was not set. This enables us not to rely on defaults.

    #RY. decorator wraps inner to have same metadata as func
    @functools.wraps(func)
    def inner(*args, **kwargs):
        # Re-prase arguments (I know! How wasteful...) to determine which ones were actually supplied
        program = Program()
        init_pc = ParserContext(args=program.core_args() + program.task_args())
        #RY. parse commandline arguments in context of init_pc
        core = Parser(initial=init_pc, ignore_unknown=True).parse_argv(sys.argv[1:])
        pc = ParserContext(name=func.__name__, args=Task(func).get_arguments())
        p = Parser(initial=init_pc, contexts=(pc,)).parse_argv(core.unparsed)
        core_args, task_args, *_ = p

        if len(args) != 1 or not isinstance(args[0], Context):
            raise RuntimeError(
                "Expected `*args` to only contain a `Context` object as all other "
                "positional arguments should be passed in as keyword arguments. "
                f"Instead got arguments: {args}."
            )
        if task_args.as_kwargs != kwargs:
            raise RuntimeError(
                f"Error while re-parsing arguments. Invoke got {kwargs}, but {task_args.as_kwargs}"
                f"was re-parsed when determining which args were set by user."
            )

        # Since positional args are passed as kwargs (Parameter.POSITIONAL_OR_KEYWORD), make sure to add
        # them back to `supplied_kwargs` otherwise we accidentally filter them out!
        supplied_kwargs = set(
            arg.attr_name or arg.name for arg in task_args.args.values() if arg.got_value and not arg.positional
        )
        supplied_kwargs |= set(arg.attr_name or arg.name for arg in task_args.args.values() if arg.positional)
        kwargs = {k: v for k, v in kwargs.items() if k in supplied_kwargs}
        return func(*args, **kwargs)

    return inner


def add_args_to_signature(*arg_names, replace_kw_only=True, **kwarg_names):
    """Decorator that extends a function's signature by adding new positional or keyword argumetns.

    Args:
        *arg_names: Names of positional arguments to add to function signature
        replace_kw_only: Flag to downgrade keyword only arguments to for compatibility defaults to true.

    :returns:
        Decorated function with extended signature.

    """

    def inner(func: Callable) -> Callable:
        signature = inspect.signature(func)
        params = list(signature.parameters.values())

        if inspect.Parameter.VAR_POSITIONAL not in (p.kind for p in params) and arg_names:
            raise ValueError(
                "Cannot add new arguments to signature if original signature doesn't accept *args. "
                "Otherwise they'll have nowhere to go!"
            )
        if inspect.Parameter.VAR_KEYWORD not in (p.kind for p in params) and kwarg_names:
            raise ValueError(
                "Cannot add new keyword arguments to signature if original signature doesn't accept *kwargs. "
                "Otherwise they'll have nowhere to go!"
            )

        #RY. Create Parameter objects for args and kwargs
        args = [inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in arg_names]
        kwargs = [
            inspect.Parameter(kwarg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
            for kwarg, default in kwarg_names.items()
        ]

        # We sort based on param type to ensure we have pos-only, pos-or-kw, var-pos, kw-only, var-kw.
        # Python uses timsort, which is stable, so we are ensured that params of the same
        # type won't be shuffled around. Specifically, the context object will remain first.
        params = sorted(params + args + kwargs, key=lambda p: p.kind)

        # Keyword only args aren't always well supported, downgrade them if necessary.
        params = [
            inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=p.default)
            if p.kind == inspect.Parameter.KEYWORD_ONLY and replace_kw_only
            else p
            for p in params
        ]

        # Yes, we need to resort here because the above step might have changed the order of things.
        # Specifically, kw-only args should always come after kw-or-pos ones, but we can't know that
        # after converting them above. So we need to sort them in their correct spots first, then
        # change their types and resort.
        func.__signature__ = signature.replace(parameters=sorted(params, key=lambda p: p.kind))
        return functools.wraps(func)(func)

    return inner


def remove_args_from_signature(*arg_or_kwarg_names, remove_var_args=False, remove_var_kwargs=False):
    """Decorator that selectively removes specified arguments from function signature.

    Args:
        *arg_or_kwarg_names: Names of positional or keyword arguments to be removed.
        remove_var_args: Flag to remove variable length args.
        remove_var_kwargs: Flag to remove keyword length args.

    :returns:
        Decorated argument with specified arguments removed
    """

    def inner(func: Callable) -> Callable:
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        #RY. filter out parameters matching specified params
        params = [p for p in params if p.name not in set(arg_or_kwarg_names)]

        # Optionally remove wildcard *args, **kwargs
        if remove_var_args:
            params = [p for p in params if p.kind != inspect.Parameter.VAR_POSITIONAL]
        if remove_var_kwargs:
            params = [p for p in params if p.kind != inspect.Parameter.VAR_KEYWORD]

        #RY. modify funciton signature and assigned back to func
        func.__signature__ = signature.replace(parameters=params)
        return functools.wraps(func)(func)

    return inner


def modify_signature(
    *arg_names,
    remove_var_args=False,
    remove_var_kwargs=False,
    remove_args_or_kwargs=(),
    add_context=False,
    only_pass_if_present=False,
    replace_kw_only=True,
    docstring=None,
    **kwarg_names,
):
    """Dynamically modify function signature  by combining add args, remove args, ignore context,  pass if present.

    Args:
        *arg_names: Arg names to be added.
        remove_var_args: Flag to remove variable length args. Defaults to false.
        remove_var_kwargs: Flag to remove variable length kwargs. Defaults to false.
        remove_args_or_kwargs: Args ro kwarg names to be removed.
        add_context: Flag to add context argument to funciton signature.
        only_pass_if_present: Flag to only pass kwargs specified in CLI. Defaults to false.
        replace_kw_only: Flag to downgrade kwargs to args or kwargs for compatibility. Defaults to true.
        doctring: Docstring. Defaults to none.
        **kwargs_names: Kwarg names to be added.
    
    :returns:
        Returns decorated function with specified modifications.
    """

    def inner(func: Callable) -> Callable:
        # We need to remove args/kwargs first in case new (kw)args
        # by the same names are added later, but we cannot yet remove
        # var-(kw)args because without them we can't add new args!
        if remove_args_or_kwargs:
            func = remove_args_from_signature(
                *remove_args_or_kwargs,
            )(func)

        if arg_names or kwarg_names or replace_kw_only:
            func = add_args_to_signature(
                *arg_names,
                replace_kw_only=replace_kw_only,
                **kwarg_names,
            )(func)

        if remove_var_args or remove_var_kwargs:
            func = remove_args_from_signature(
                remove_var_args=remove_var_args,
                remove_var_kwargs=remove_var_kwargs,
            )(func)

        if add_context:
            func = ignore_context(func)

        if only_pass_if_present:
            func = pass_if_present(func)

        if docstring:
            func.__doc__ = inspect.cleandoc(docstring)

        return func

    return inner


class StreamWatcherTqdmPbar:
    """Watch the streams generated by a `c.run` call and update a progress bar when
    something in the stream matches the regex `pattern`. If the pattern has a named
    group with the name 'n' (i.e: something like '(?P<n>\d+)'), we set the progress
    bar to that value, any other match simply increments the progressbar by a tick.
    """

    def __init__(self, pattern, description="Processing...", on_match=None, **pbar_kwargs):
        self.pattern = pattern
        self.description = description
        self.on_match = on_match or {}
        self.pbar_kwargs = pbar_kwargs
        self.pbar = None
        self.index = 0

    def __enter__(self):
        self.pbar = tqdm(**self.pbar_kwargs)
        self.pbar.set_description(self.description)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()

    def submit(self, stream):
        # Only look at stream contents we haven't seen yet, to avoid dupes.
        new_ = stream[self.index :]

        # Allow for arbitrary callbacks if a specific match is found
        for pattern, on_match_callback in self.on_match.items():
            matches = re.search(pattern, new_, re.DOTALL)

            if matches:
                on_match_callback(matches)

        # Search, across lines if necessary
        matches = re.search(self.pattern, new_, re.DOTALL)

        # Update seek index and progress bar if we've matched
        if matches:
            if n := matches.groupdict().get("n"):
                self.pbar.n = int(n)
                self.pbar.last_print_n = int(n)
            else:
                self.pbar.update(1)
            self.pbar.refresh()
            self.index += len(new_)
        return []
