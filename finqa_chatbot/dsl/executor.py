"""DSL program executor — evaluates tokenized programs against tables."""

from __future__ import annotations

from .operations import ALL_OPS


def str_to_num(text: str) -> float | str:
    """Convert a string to a number, handling percentages and constants."""
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                return float(text) / 100.0
            except ValueError:
                return "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            return float(text)
        else:
            return "n/a"


def process_row(row_in: list[str]) -> list[float] | str:
    """Convert a table row of strings to floats."""
    row_out = []
    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()
        num = str_to_num(num)
        if num == "n/a":
            return "n/a"
        row_out.append(num)
    return row_out


def eval_program(
    program: list[str], table: list[list[str]]
) -> tuple[int, float | str]:
    """Execute a tokenized program and return ``(invalid_flag, result)``.

    The token list must end with ``"EOF"``.
    """
    invalid_flag = 0
    this_res: float | str = "n/a"

    try:
        program = program[:-1]  # remove EOF
        # structural validation
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program_str = "|".join(program)
        steps = program_str.split(")")[:-1]
        res_dict: dict[int, float | str] = {}

        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break

            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op in ("add", "subtract", "multiply", "divide", "exp", "greater"):
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break
                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break

                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"
                res_dict[ind] = this_res

            elif "table" in op:
                table_dict: dict[str, list[str]] = {}
                for row in table:
                    table_dict[row[0]] = row[1:]

                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)

                if num_row == "n/a":
                    invalid_flag = 1
                    break
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)
                res_dict[ind] = this_res

        if this_res != "yes" and this_res != "no" and this_res != "n/a":
            this_res = round(this_res, 5)
    except Exception:
        invalid_flag = 1

    return invalid_flag, this_res
