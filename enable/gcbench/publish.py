# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import math
import os

_DOC_TEMPLATE = """
<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<title>Graphics Context Benchmark Results</title>
</head>
<body>

<style>
  table, th, td {{
    padding: 4px;
    border: 1px solid gray;
    border-collapse: collapse;
  }}
  th {{
    text-align: left;
  }}
</style>
<h3>Kiva Backend Benchmark Results</h3>
<p>
All results are shown relative to the kiva.agg backend. Numbers less than 1.0
indicate a slower result and numbers greater than 1.0 indicate a faster result.
</p>
{content}
</body>
</html>
"""
_TABLE_TEMPLATE = """
<table>
<tr>
{headers}
</tr>
{rows}
</table>
"""


def publish(results, outdir):
    """ Write the test results out as a simple webpage.
    """
    backends = list(results)
    functions = {}
    for bend in backends:
        for func, stats in results[bend].items():
            value = stats["mean"] if stats is not None else math.nan
            functions.setdefault(func, {})[bend] = value

    # Scale timing values relative to a "baseline" backend implementation
    for name in functions.keys():
        functions[name] = _format_stats(functions[name], "kiva.agg")

    # Build some table data
    headers = ["Draw Function"] + backends
    headers = "\n".join(f"<th>{head}</th>" for head in headers)
    rows = [
        [f"<td>{name}</td>"] + [f"<td>{stats[bend]}</td>" for bend in backends]
        for name, stats in functions.items()
    ]
    rows = ["".join(row) for row in rows]
    rows = "\n".join(f"<tr>{row}</tr>" for row in rows)
    table = _TABLE_TEMPLATE.format(headers=headers, rows=rows)

    path = os.path.join(outdir, "index.html")
    with open(path, "w") as fp:
        fp.write(_DOC_TEMPLATE.format(content=table))


def _format_stats(function, baseline):
    basevalue = function[baseline]
    formatted = {}
    for name, value in function.items():
        if value is math.nan:
            formatted[name] = "X"
        else:
            relvalue = basevalue / value
            formatted[name] = f"{relvalue:0.2f}"

    return formatted
