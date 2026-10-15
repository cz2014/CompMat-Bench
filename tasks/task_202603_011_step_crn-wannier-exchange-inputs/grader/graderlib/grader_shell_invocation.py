"""Shell-script primitives for gen_input subtasks whose submission includes a
command (a `run.sh`) rather than only structured input files.

Scope: locate one program's invocation inside a shell script and return its
options as data. The per-subtask grader then writes REQUIRED rules on those
options exactly as it does on INCAR tags. No execution ever happens: the file
is read as text.

What it handles, because a correct submission may legitimately be written any
of these ways:

  * backslash line continuations (a long command spread over many lines);
  * `#` comments, quoting, and `$(...)`/backtick command substitution inside a
    quoted argument (tokenised, never run);
  * a launcher prefix (`mpirun -np 8 wann2J.py ...`, `python3 wann2J.py ...`)
    and a path prefix (`/opt/bin/wann2J.py`);
  * `--opt value`, `--opt=value`, multi-valued `--opt v1 v2 v3`, and negative
    numeric values (`--emin -14.0`) which must not be read as a new option;
  * a literal shell assignment used later as `$VAR` / `${VAR}`
    (`EF=9.6310` ... `--efermi "$EF"`), which is expanded before the value is
    handed back. A value produced by command substitution stays unexpanded, so
    a rule that needs a number fails it with a clear reason rather than
    guessing.

Requires: stdlib only.
"""
from __future__ import annotations

import re
import math
import shlex
from pathlib import Path

from grader_numeric import GradeError

#: tokens that terminate the command (redirection, pipes, separators)
_STOP = {">", ">>", "<", "|", "||", "&&", ";", "&", "2>", "2>&1", "|&"}

_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_VARREF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _join_continuations(text):
    """Fold backslash-newline continuations into single logical lines."""
    return re.sub(r"\\\s*\n", " ", text)


def _expand(token, variables):
    return _VARREF.sub(lambda m: variables.get(m.group(1) or m.group(2), m.group(0)), token)


def _shell_tokens(line, variables):
    """Bounded lexical shell scan. Preserve single quotes; never execute code."""
    tokens, parts = [], []
    quote = None
    i = 0
    started = False
    while i < len(line):
        c = line[i]
        if c == "\\" and quote != "'" and i+1 < len(line):
            parts.append(line[i+1]); started = True; i += 2; continue
        if c in ("'", '"'):
            if quote is None:
                quote = c; started = True; i += 1; continue
            if quote == c:
                quote = None; i += 1; continue
        if quote is None and (c.isspace() or c in ";|&<>"):
            if started:
                tokens.append("".join(parts)); parts = []; started = False
            if not c.isspace():
                tokens.append(c)
            i += 1; continue
        if quote is None and c == "#" and not started:
            break
        if c == "$" and quote != "'":
            match = _VARREF.match(line, i)
            if match:
                parts.append(variables.get(match.group(1) or match.group(2), match.group(0)))
                started = True; i = match.end(); continue
        parts.append(c); started = True; i += 1
    if quote:
        raise GradeError("run script: unterminated shell quote")
    if started:
        tokens.append("".join(parts))
    return tokens


def _simple_commands(text):
    """Split unquoted semicolons before expansion, retaining shell quoting.

    Substitution bodies remain opaque. Control operators are deliberately not
    interpreted: their conditional/subshell execution cannot be established by
    this literal-command reader.
    """
    for line in _join_continuations(text).splitlines():
        start, i, depth = 0, 0, 0
        quote = None
        backtick = False
        while i < len(line):
            c = line[i]
            if c == "\\" and quote != "'":
                i += 2
                continue
            if c == "`" and quote != "'":
                backtick = not backtick
            elif not backtick:
                if c in ("'", '"') and not depth:
                    if quote is None:
                        quote = c
                    elif quote == c:
                        quote = None
                elif quote != "'" and line[i:i+2] == "$(":
                    depth += 1
                    i += 1
                elif depth and c == "(":
                    depth += 1
                elif depth and c == ")":
                    depth -= 1
                elif quote is None and not depth:
                    if c == "#" and (i == 0 or line[i-1].isspace() or line[i-1] == ";"):
                        line = line[:i]
                        break
                    if c == ";":
                        yield line[start:i]
                        start = i+1
                    elif c in "|&" and not (c == "&" and i and line[i-1] in "><"):
                        raise GradeError("run script: unsupported shell control operator")
            i += 1
        if quote or depth or backtick:
            raise GradeError("run script: unterminated shell quote or substitution")
        yield line[start:]


def _assignments(tokens, variables):
    """Return assignment-only state and prefix length; arguments stay pre-command."""
    local = dict(variables)
    nassign = 0
    for token in tokens:
        match = _ASSIGN.match(token)
        if not match:
            break
        local[match.group(1)] = match.group(2)
        nassign += 1
    return local, nassign


def literal_assignments(text):
    variables = {}
    for command in _simple_commands(text):
        tokens = _shell_tokens(command, variables)
        if tokens and tokens[0] == "export":
            tokens = tokens[1:]
        local, count = _assignments(tokens, variables)
        if count == len(tokens):
            variables = local
    return variables


def _program_index(tokens, program):
    """Accept a command position, optionally behind common static launchers."""
    i = 0
    while i < len(tokens):
        name = Path(tokens[i]).name
        if name == program:
            return i
        if name in ("exec", "env"):
            i += 1
            while i < len(tokens) and _ASSIGN.match(tokens[i]):
                i += 1
        elif re.fullmatch(r"python(?:[23](?:\.\d+)?)?", name):
            i += 1
            while i < len(tokens) and tokens[i] in ("-u", "-B", "-E", "-s"):
                i += 1
        elif name in ("mpirun", "mpiexec", "srun"):
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                opt = tokens[i]; i += 1
                if opt in ("-np", "-n", "--ntasks", "--nodes", "-N", "--cpus-per-task"):
                    i += 1
                elif "=" not in opt:
                    return None
        else:
            return None
    return None


def find_invocation(path, program):
    """Read actual static command positions with assignments effective at that line.

    This is a bounded input rubric, not a shell interpreter. Echoed commands,
    unresolved substitutions and single-quoted variables are never executable
    numeric evidence. Unsupported control flow is not guessed.
    """
    variables, candidates = {}, []
    for command in _simple_commands(Path(path).read_text(errors="replace")):
        tokens = _shell_tokens(command, variables)
        if not tokens:
            continue
        if tokens[0] in ("if", "for", "while", "until", "case", "function", "then", "else", "elif", "do") or tokens[0].endswith("()"):
            raise GradeError("run script: dynamic control flow needs a directly inspectable invocation")
        if tokens[0] == "export":
            tokens = tokens[1:]
        local, nassign = _assignments(tokens, variables)
        if nassign == len(tokens):
            variables = local
            continue
        # Prefix assignments affect the child's environment, not the shell's
        # expansion of this command's arguments or subsequent commands.
        tokens = tokens[nassign:]
        for i, token in enumerate(tokens):
            if token in (">", "<"):
                tokens = tokens[:i]
                break
        idx = _program_index(tokens, program)
        if idx is not None:
            candidates.append((tokens[idx:], dict(variables)))
    if not candidates:
        raise GradeError(f"run script: no actual `{program}` invocation found")
    return candidates[-1]


def _is_value(token):
    """A token is a value (not a new option) when it does not start with '-',
    or when it is a negative number such as -14.0."""
    if not token.startswith("-"):
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_options(argv):
    """{option name without dashes: [value, ...]} for the argv of one command.

    A flag with no value gets an empty list. `--opt=value` and `--opt value`
    are equivalent. Positional arguments are returned under the key
    `_positional`.
    """
    opts, positional = {}, []
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and not _is_value(tok):
            if "=" in tok:
                name, value = tok.lstrip("-").split("=", 1)
                opts[name] = [value]
                i += 1
                continue
            name = tok.lstrip("-")
            values = []
            i += 1
            while i < len(argv) and _is_value(argv[i]):
                values.append(argv[i])
                i += 1
            opts[name] = values
            continue
        positional.append(tok)
        i += 1
    opts["_positional"] = positional
    return opts


def option_floats(opts, name, count=None):
    """Numeric values of an option. Returns None when the option is absent;
    raises GradeError when a value is present but not a number (an unexpanded
    `$(...)` substitution lands here)."""
    if name not in opts:
        return None
    vals = opts[name]
    out = []
    for v in vals:
        try:
            number = float(v)
            if not math.isfinite(number):
                raise ValueError("nonfinite")
            out.append(number)
        except ValueError:
            raise GradeError(f"--{name} value '{v}' is not a number")
    if count is not None and len(out) != count:
        raise GradeError(f"--{name} needs {count} values, got {len(out)}: {vals}")
    return out


def option_str(opts, name):
    """Single string value of an option, or None when absent."""
    if name not in opts or not opts[name]:
        return None
    return opts[name][0]
