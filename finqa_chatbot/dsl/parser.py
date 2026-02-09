"""DSL program parsing — string to token list conversion."""

from __future__ import annotations

import re

from .operations import ALL_OPS


def unnest_program(program_str: str) -> str:
    """Convert nested format into sequential format.

    Example: ``multiply(divide(59.1, 98.0), const_100)``
    becomes: ``divide(59.1, 98.0), multiply(#0, const_100)``
    """
    program_str = program_str.strip()
    all_ops_pattern = "|".join(ALL_OPS)
    nested_pattern = re.compile(
        rf'({all_ops_pattern})\(.*({all_ops_pattern})\('
    )
    if not nested_pattern.search(program_str):
        return program_str

    steps: list[str] = []

    def parse_expr(s: str, pos: int = 0) -> tuple[str, int]:
        op_match = re.match(r'(' + all_ops_pattern + r')\(', s[pos:])
        if op_match:
            op = op_match.group(1)
            pos += len(op_match.group(0))
            arg1, pos = parse_expr(s, pos)
            while pos < len(s) and s[pos] in ', ':
                pos += 1
            arg2, pos = parse_expr(s, pos)
            while pos < len(s) and s[pos] in ') ':
                if s[pos] == ')':
                    pos += 1
                    break
                pos += 1
            step_idx = len(steps)
            steps.append(f"{op}({arg1}, {arg2})")
            return f"#{step_idx}", pos
        else:
            end = pos
            paren_depth = 0
            while end < len(s):
                if s[end] == '(':
                    paren_depth += 1
                elif s[end] == ')':
                    if paren_depth == 0:
                        break
                    paren_depth -= 1
                elif s[end] == ',' and paren_depth == 0:
                    break
                end += 1
            literal = s[pos:end].strip()
            return literal, end

    try:
        remaining = program_str
        while remaining.strip():
            remaining = remaining.strip().lstrip(',').strip()
            if not remaining:
                break
            op_match = re.match(r'(' + all_ops_pattern + r')\(', remaining)
            if op_match:
                _ref, new_pos = parse_expr(remaining, 0)
                remaining = remaining[new_pos:].strip().lstrip(',').strip()
            else:
                break
        if steps:
            return ", ".join(steps)
    except Exception:
        pass

    return program_str


def parse_program_to_tokens(program_str: str) -> list[str]:
    """Convert a program string to a token list for evaluation.

    Returns tokens in the format expected by ``eval_program``,
    ending with ``"EOF"``.
    """
    program_str = program_str.strip()
    # Remove markdown formatting
    program_str = program_str.strip("`").strip()
    if program_str.startswith("Program:"):
        program_str = program_str[len("Program:"):].strip()

    # Find the line that contains an operation
    lines = program_str.split('\n')
    for line in lines:
        line = line.strip()
        if line and any(op + '(' in line for op in ALL_OPS):
            program_str = line
            break

    # Unnest if needed
    program_str = unnest_program(program_str)

    # Tokenize: op(, arg1, arg2, )
    tokens: list[str] = []
    original_program = program_str.split(', ')
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    tokens.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                tokens.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            tokens.append(cur_tok)
    tokens.append('EOF')
    return tokens
