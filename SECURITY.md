# Security Policy

Thank you for helping keep Easy-EO and the people who depend on it safe.

## Supported versions

Easy-EO is pre-1.0, so fixes land on the latest release rather than being
backported. If you are running an older version, the first step is usually to
upgrade.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities in public issues, pull
requests, or discussions.** A public report tells everyone about the problem,
including people who would use it, before there is a fix to upgrade to.

Instead, report it privately through GitHub:

**https://github.com/Tommy-Burns/easy-eo/security/advisories/new**

That opens a private advisory visible only to you and the maintainer. If you
cannot use it for any reason, open a public issue containing no detail — just
asking for a private contact — and you will be given one.

A useful report usually includes:

- the version of Easy-EO, Python, rasterio and GDAL involved
  (`python -c "import eeo; eeo.show_versions()"` prints all of them),
- what an attacker gains, and what they need in order to attempt it,
- the smallest reproduction you can manage — a short script, and a raster or
  vector file if the problem depends on one.

## What to expect

Easy-EO is maintained by a very few people, so these are honest targets rather than a
commercial support agreement:

- **Acknowledgement within 7 days.** If you have heard nothing after that,
  please assume the notification was missed and comment on the advisory.
- **An initial assessment within 14 days**, saying whether the report is
  accepted, needs more information, or is out of scope, and why.
- **A fix released as soon as it is ready.** Serious vulnerabilities are
  released on their own rather than waiting for the next planned version.

You will be credited in the advisory and the changelog unless you would rather
not be.

## Scope

In scope:

- the `eeo` package itself,
- the sample-data download path (`eeo.datasets`), including its checksum
  verification,
- the build and release workflows in `.github/workflows/`, and anything that
  could put unintended content into a published PyPI or conda-forge artifact.

Out of scope, though still worth telling us about:

- **Vulnerabilities in dependencies.** rasterio, GDAL, NumPy, GeoPandas and
  Matplotlib are separate projects with their own security processes, and
  reports belong with them. Do tell us if Easy-EO's particular use of one makes
  an upstream issue materially worse, or if a dependency floor in
  `pyproject.toml` permits a version known to be vulnerable.
- Anything requiring an attacker who already controls the machine or the Python
  environment. Someone who can run arbitrary Python does not need Easy-EO.

## A note on untrusted raster files

Easy-EO does not implement raster parsers. Opening a file hands it to rasterio
and, beneath that, to GDAL, which is where the format parsing and therefore the
real trust boundary lives. Treat untrusted rasters with the same caution you
would give any GDAL-based tool: process them in a sandbox, and keep GDAL
current. A parsing crash triggered through Easy-EO is normally a GDAL issue —
report it there — but tell us as well if Easy-EO turns something harmless
upstream into something worse.

## Disclosure

Disclosure is coordinated. Once a fix is available, the advisory is published
with credit to the reporter, the fix is described in `CHANGELOG.md`, and a CVE
is requested through GitHub where the issue warrants one. If a report is
declined as out of scope, you are free to disclose it however you see fit.
