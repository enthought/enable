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
    background: #eee;
    border: 1px solid gray;
    border-collapse: collapse;
  }}
  th {{
    text-align: center;
  }}
  td.valid,td.invalid,td.skipped {{
    text-align: center;
    vertical-align: center;
  }}
  td.valid {{
    background: lightgreen;
  }}
  td.invalid {{
    background: lightpink;
  }}
  td.skipped {{
    background: inherit;
  }}
</style>
<h3>Kiva Backend Benchmark Results</h3>
<p>
All results are shown relative to the kiva.agg backend. Numbers less than 1.0
indicate a slower result and numbers greater than 1.0 indicate a faster result.
<br><br>
For backends that aren't timed:<br>
"\N{HEAVY CHECK MARK}" indicates a successful run<br>
"\N{HEAVY BALLOT X}" indicates a failed run<br>
"\N{HEAVY MINUS SIGN}" indicates a skipped run<br>
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
  table, td {{
    padding: 4px;
    border: 1px solid gray;
    border-collapse: collapse;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    text-align: center;
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
<tr>{headers}</tr>
{rows}
</table>
"""


def publish(results, outdir):
    """ Write the test results out as a simple webpage.
    """
    backends = []
    benchmarks = {}

    # Transpose the results so that they're accesible by benchmark.
    for btype, backend_results in results.items():
        backends.extend(list(backend_results))
        for bend in backend_results:
            for benchmark_name, res in backend_results[bend].items():
                benchmarks.setdefault(benchmark_name, {})[bend] = res

    # Convert each benchmark into an output comparison page and a row for the
    # comparison table.
    comparisons = {}
    for benchmark_name, benchmark_results in benchmarks.items():
        _build_output_comparison_page(
            benchmark_name, benchmark_results, outdir
        )
        # Compare each result to the "kiva.agg" result
        baseline = benchmark_results["kiva.agg"]
        comparisons[benchmark_name] = {
            name: result.compare_to(baseline)
            for name, result in benchmark_results.items()
        }

    # Fill out the comparison table and write the summary index
    comparison_table = _build_comparison_table(backends, comparisons)
    path = os.path.join(outdir, "index.html")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_INDEX_TEMPLATE.format(comparison_table=comparison_table))


def _build_comparison_table(backends, comparisons):
    """ Build some table data for comparison of backend performance timings.
    """
    # Headers
    headers = ["Draw Function"] + backends
    headers = "\n".join(_th(head) for head in headers)

    # Build the rows
    rows = []
    for benchmark_name, comparisons in comparisons.items():
        # Start the row off with the name of the benchmark
        # Link to the benchmark output comparison page
        row = [_td(_link(f"{benchmark_name}.html", benchmark_name))]

        # Add column entries for the BenchComparisons, ordered by backend
        for bend in backends:
            comp = comparisons[bend]
            row.append(f'<td class="{comp.css_class}">{comp.value}</td>')

        # Concat all the columns into a single table row
        rows.append(_tr("".join(row)))
    rows = "\n".join(rows)

    # Smash it all together in the template
    return _TABLE_TEMPLATE.format(headers=headers, rows=rows)


def _build_output_comparison_page(benchmark_name, backend_results, outdir):
    """ Build a page which shows backend outputs next to each other.
    """
    # Headers
    headers = ("Backend", "Output", "Timing")
    headers = "".join(_th(name) for name in headers)

    # Build the rows
    rows = []
    for backend_name, result in backend_results.items():
        # If no file was output, skip
        if not result.output:
            continue

        # A row is [Backend | Output | Timing]
        output = _format_output(result)
        timing = _format_timing(result)
        rows.append(_tr(f"{_td(backend_name)}{_td(output)}{_td(timing)}"))
    rows = "\n".join(rows)

    table = _TABLE_TEMPLATE.format(headers=headers, rows=rows)
    content = _IMAGE_PAGE_TEMPLATE.format(
        benchmark_name=benchmark_name,
        image_table=table,
    )
    path = os.path.join(outdir, f"{benchmark_name}.html")
    with open(path, "w") as fp:
        fp.write(content)


def _format_output(result):
    """ Convert the output from a single benchmark run into an image embed or
    link.
    """
    if result.output_format in (".png", ".svg"):
        return _img(result.output, *result.output_size)
    else:
        return _link(result.output, "download")


def _format_timing(result):
    """ Convert timing stats for a single benchmark run into a table.
    """
    if result.timing is None:
        return ""
    return result.timing.to_html()


# HTML utils
def _img(src, width, height):
    return f'<img src="{src}" width="{width}" height="{height}"/>'


def _link(target, text):
    return f'<a href="{target}">{text}</a>'


def _td(data, **attrs):
    return f"<td>{data}</td>"


def _th(data):
    return f"<th>{data}</th>"


def _tr(data):
    return f"<tr>{data}</tr>"
