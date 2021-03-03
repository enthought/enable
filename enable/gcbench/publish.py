# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import os

_INDEX_TEMPLATE = """
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
  td.valid {{
    background: lightgreen;
  }}
  td.invalid {{
    background: lightpink;
  }}
    td.skipped {{
  }}
</style>
<h3>Kiva Backend Benchmark Results</h3>
<p>
All results are shown relative to the kiva.agg backend. Numbers less than 1.0
indicate a slower result and numbers greater than 1.0 indicate a faster result.
</p>
{comparison_table}
</body>
</html>
"""
_IMAGE_PAGE_TEMPLATE = """
<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<title>{benchmark_name} Outputs</title>
</head>
<body>
<style>
  table, th, td {{
    padding: 4px;
    border: 1px solid gray;
    border-collapse: collapse;
    text-align: left;
  }}
</style>
<p>
Results for the "{benchmark_name}" benchmark. All times are in milliseconds.
</p>
{image_table}
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
    backends = []
    functions = {}

    # Transpose the results so that they're accesible by function.
    for btype, backend_results in results.items():
        backends.extend(list(backend_results))
        for bend in backend_results:
            for name, res in backend_results[bend].items():
                functions.setdefault(name, {})[bend] = res

    comparisons = {}
    for name, results in functions.items():
        _build_function_page(name, results, outdir)
        # Scale timing values relative to the "kiva.agg" backend implementation
        comparisons[name] = _format_benchmark(results, "kiva.agg")

    comparison_table = _build_comparison_table(backends, comparisons)
    path = os.path.join(outdir, "index.html")
    with open(path, "w") as fp:
        fp.write(_INDEX_TEMPLATE.format(comparison_table=comparison_table))


def _build_comparison_table(backends, comparisons):
    """ Build some table data for comparison of backend performance timings.
    """
    # All the row data
    rows = []
    for name, stats in comparisons.items():
        # Start the row off with the name of the function
        # Link to the table of images created by each backend
        link = f'<a href="{name}.html">'
        row = [f"<td>{link}{name}</a></td>"]
        for bend in backends:
            # Each backend stat includes a CSS class for table styling
            stat, klass = stats[bend]
            row.append(f'<td class="{klass}">{stat}</td>')
        # Concat all the <td>'s into a single string
        rows.append("".join(row))
    # Concat all the <tr>'s into a multiline string.
    rows = "\n".join(f"<tr>{row}</tr>" for row in rows)

    # Headers
    headers = ["Draw Function"] + backends
    headers = "\n".join(f"<th>{head}</th>" for head in headers)

    # Smash it all together in the template
    return _TABLE_TEMPLATE.format(headers=headers, rows=rows)


def _build_function_page(benchmark_name, results, outdir):
    """ Build a page which shows backend outputs next to each other.
    """
    # Build the rows
    backends = []
    output_tds, stat_tds = "", ""
    for backend_name, result in results.items():
        if result is None or "skip" in result:
            continue

        backends.append(backend_name)
        output_tds += f"<td>{_format_output(result)}</td>"
        stat_tds += f"<td>{_format_stats(result['times'])}</td>"

    rows = f"<tr>{output_tds}</tr>\n<tr>{stat_tds}</tr>"

    # Headers
    headers = "\n".join(f"<th>{name}</th>" for name in backends)

    table = _TABLE_TEMPLATE.format(headers=headers, rows=rows)
    content = _IMAGE_PAGE_TEMPLATE.format(
        benchmark_name=benchmark_name,
        image_table=table,
    )
    path = os.path.join(outdir, f"{benchmark_name}.html")
    with open(path, "w") as fp:
        fp.write(content)


def _format_benchmark(results, baseline):
    """ Convert stats for backend benchmark runs into data for a table row.
    """
    basevalue = results[baseline]["times"]["mean"]
    formatted = {}
    for name, result in results.items():
        if result is not None:
            stats = result.get("times", {})
            if stats:
                relvalue = basevalue / stats["mean"]
                formatted[name] = (f"{relvalue:0.2f}", "valid")
            else:
                if "skip" in result:
                    # Benchmark was skipped
                    formatted[name] = ("\N{HEAVY MINUS SIGN}", "skipped")
                else:
                    # No times, but the backend succeeded
                    formatted[name] = ("\N{HEAVY CHECK MARK}", "valid")
        else:
            formatted[name] = ("\N{HEAVY BALLOT X}", "invalid")

    return formatted


def _format_output(result):
    """ Convert the output from a single benchmark run into an image embed or
    link.
    """
    if result["format"] in ("png", "svg"):
        return f'<img src="{result["filename"]}" />'
    else:
        return f'<a href="{result["filename"]}">download</a>'


def _format_stats(stats):
    """ Convert timing stats for a single benchmark run into a table.
    """
    rows = [
        f"<tr><td>{key.capitalize()}</td><td>{value:0.4f}</td></tr>"
        for key, value in stats.items()
    ]
    rows = "\n".join(rows)
    return f"<p>Timings:</p><table>{rows}</table>"
