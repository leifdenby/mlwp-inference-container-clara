"""
Set the options of a command line interface from environment variables.

The convention, for the entrypoints of MLWP inference containers: each model
has an environment variable prefix (e.g. `ANNA`), and every option
`--some-option` of its entrypoint can also be set with the variable
`{PREFIX}_SOME_OPTION`. An option given on the command line wins over its
variable, which wins over the option's default:

    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-duration", default="PT18H")
    envargs.add_env_help(parser, "ANNA")
    args, sources = envargs.parse_args(parser, "ANNA")
    for line in envargs.config_lines(parser, args, sources, "ANNA"):
        logger.info(line)

- The variables are turned into command line arguments placed before the
  real ones, so argparse checks their types and choices, and a required
  option can come from a variable alone.
- A flag (e.g. `store_true`) takes `true`/`false` (also `1`/`0`,
  `yes`/`no`), and an option taking several values a space-separated list.
- A set variable with the prefix that matches no option is an error (a
  misspelling would otherwise have no effect), with a suggestion of close
  names. Variables without the prefix are ignored.
- An empty variable counts as unset.
- Only options are supported, no positional arguments or subcommands.

This module only uses the standard library, so it can be copied to (or
extracted for) other inference containers.
"""
import argparse
import copy
import difflib
import os
import shlex
import sys

TRUE_VALUES = ("true", "1", "yes")
FALSE_VALUES = ("false", "0", "no")

SOURCE_CLI, SOURCE_ENV, SOURCE_DEFAULT = "cli", "env", "default"


def env_var_name(prefix, option):
    """`("ANNA", "--forecast-duration")` -> `CLARA_FORECAST_DURATION`."""
    return f"{prefix}_{option.lstrip('-').upper().replace('-', '_')}"


def _options(parser):
    """(action, long option) for each option of `parser`, except --help."""
    options = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        long_options = [o for o in action.option_strings if o.startswith("--")]
        if not long_options:
            raise ValueError(
                f"envargs only supports long options, not {action.dest!r} "
                "(a positional argument, subcommand or short-only option)"
            )
        options.append((action, long_options[0]))
    return options


def env_var_names(parser, prefix):
    """The environment variable name of each option, by `dest`."""
    return {
        action.dest: env_var_name(prefix, option)
        for action, option in _options(parser)
    }


def add_env_help(parser, prefix):
    """Add `[env: {PREFIX}_OPTION]` to the help of each option."""
    for action, option in _options(parser):
        if action.help == argparse.SUPPRESS:
            continue
        env_help = f"[env: {env_var_name(prefix, option)}]"
        action.help = f"{action.help} {env_help}" if action.help else env_help
    convention = (
        f"Every option --some-option can also be set with the environment "
        f"variable {prefix}_SOME_OPTION; an option on the command line wins "
        f"over its variable."
    )
    parser.epilog = (
        f"{parser.epilog}\n\n{convention}" if parser.epilog else convention
    )


def _parse_bool(parser, name, value):
    if value.lower() in TRUE_VALUES:
        return True
    if value.lower() in FALSE_VALUES:
        return False
    parser.error(
        f"{name}={value!r}: expected one of "
        f"{', '.join(TRUE_VALUES + FALSE_VALUES)}"
    )


def _check_value(parser, action, name, value):
    """Check a value from `name` as argparse would, naming the variable."""
    if callable(action.type):
        try:
            converted = action.type(value)
        except (TypeError, ValueError, argparse.ArgumentTypeError) as ex:
            parser.error(f"{name}={value!r}: invalid value ({ex})")
    else:
        converted = value
    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(map(repr, action.choices))
        parser.error(
            f"{name}={value!r}: invalid choice (choose from {choices})"
        )


def _env_tokens(parser, action, option, name, value):
    """The command line arguments for `name={value}`."""
    if isinstance(action, argparse.BooleanOptionalAction):
        if _parse_bool(parser, name, value):
            return [option]
        return [f"--no-{option[2:]}"]
    if action.nargs == 0:
        # store_true, store_false, store_const, count: given or not
        return [option] if _parse_bool(parser, name, value) else []
    if action.nargs in (None, "?"):
        _check_value(parser, action, name, value)
        # `--option=value`, so a value starting with "-" isn't an option
        return [f"{option}={value}"]
    values = shlex.split(value)
    for v in values:
        _check_value(parser, action, name, v)
    return [option, *values]


def env_argv(parser, prefix, environ=None):
    """
    The command line arguments from the `{prefix}_*` variables in `environ`
    (default `os.environ`), and the `dest`s they set.
    """
    environ = os.environ if environ is None else environ
    names = env_var_names(parser, prefix)
    actions = {
        action.dest: (action, option) for action, option in _options(parser)
    }

    unknown = sorted(
        n
        for n in environ
        if n.startswith(f"{prefix}_") and n not in names.values()
    )
    if unknown:
        problems = []
        for n in unknown:
            close = difflib.get_close_matches(n, names.values(), n=1)
            hint = f" (did you mean {close[0]}?)" if close else ""
            problems.append(f"{n}{hint}")
        parser.error(
            f"unknown environment variable(s) {', '.join(problems)}: "
            f"{prefix}_* variables must match an option (see --help)"
        )

    argv, from_env = [], set()
    for dest, name in names.items():
        # an empty variable counts as unset (e.g. `-e NAME=` in docker)
        if environ.get(name, "") != "":
            action, option = actions[dest]
            argv += _env_tokens(parser, action, option, name, environ[name])
            from_env.add(dest)
    return argv, from_env


def _cli_dests(parser, argv):
    """The `dest`s given in `argv` (the command line alone)."""
    cli_parser = copy.deepcopy(parser)
    for action in cli_parser._actions:
        action.default = argparse.SUPPRESS
        action.required = False
    return set(vars(cli_parser.parse_args(argv)))


def parse_args(parser, prefix, argv=None, environ=None):
    """
    Parse the command line `argv` (default `sys.argv[1:]`) and the
    `{prefix}_*` variables in `environ` (default `os.environ`).

    Returns the parsed namespace, and where each option's value came from
    (by `dest`): "cli", "env" or "default".
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    from_env_argv, from_env = env_argv(parser, prefix, environ)
    args = parser.parse_args(from_env_argv + argv)

    from_cli = _cli_dests(parser, argv)
    sources = {}
    for dest in vars(args):
        if dest in from_cli:
            sources[dest] = SOURCE_CLI
        elif dest in from_env:
            sources[dest] = SOURCE_ENV
        else:
            sources[dest] = SOURCE_DEFAULT
    return args, sources


def config_lines(parser, args, sources, prefix):
    """
    The effective configuration, one line per option: its variable name,
    value and source. Don't use it for options holding secrets.
    """
    names = env_var_names(parser, prefix)
    width = max(map(len, names.values()), default=0)
    return [
        f"{name:<{width}} = {getattr(args, dest)} ({sources[dest]})"
        for dest, name in names.items()
    ]
