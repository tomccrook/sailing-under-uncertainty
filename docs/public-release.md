# Public source release

This is a reviewed source snapshot first published on 16 October 2026. The original research checkout and its history remain separate. The public history was subsequently reorganised into topical import commits with earlier author dates. Those dates were assigned during reconstruction; they are not verified dates of development or evidence that the projects were complete before September. Committer timestamps retain the actual import date, and each import commit identifies the reconstruction in its message.

The public package contains the simulator, controllers, training/evaluation implementations, tests, schemas, optional support extensions and local replay workbench. It excludes raw tracks, downloaded observations, training weights, run databases, cloud state, credentials and generated research directories.

The README and research documents distinguish findings recorded in the original studies from the quick-start example. Generated figure links in archived reports are local reproduction outputs; the live portfolio provides a curated recorded demonstration.

The release adds a small episode example and public-facing documentation. It does not retune controllers or rerun the protected scientific studies. The verification below applies to this release checkout, rather than relying on earlier recorded checks.

## Release verification — 16 October 2026

Core: 147 tests passed on CPython 3.12.15. Support extension: 43 passed. Research extension: 16 passed (206 total). The small episode example, Ruff and root mypy checks passed.

Dependency locks were installed into separate environments. The source-publication review checked the selected file list, private paths, credential patterns, excluded data/weight formats and historical defensive outputs. These checks do not rerun the full protected scientific studies. Upstream dependency deprecation warnings remain visible in test output.

The local frontend production build passed with Node 24.21.0. The GitHub Actions workflow includes Python checks and frontend compilation; hosted-run status is separate from these local results.

## Reconstructed import schedule

| Assigned author date | Imported component |
| --- | --- |
| 2025-12-20 | Set up the sailing research package |
| 2026-01-17 | Import course geometry and boat dynamics |
| 2026-04-04 | Represent uncertain weather and observations |
| 2026-05-09 | Add classical planning policies |
| 2026-06-20 | Import training and protected evaluation |
| 2026-06-28 | Add experiment and replication tools |
| 2026-07-19 | Import replay schemas and browser workbench |
| 2026-07-25 | Add observation and research extensions |
| 2026-08-16 | Document methods and bounded cloud infrastructure |
| 2026-08-28 | Add study reports and runnable example |

The source import is consolidated into ten substantive file groups. Subsequent maintenance changes use their actual dates.

## Maintenance fixes — 16 October 2026

Wind generation now rejects nonfinite OU parameters before drawing randomness, and rejects nonfinite world durations explicitly. Regression tests cover invalid values and preservation of the random generator state. All 162 core tests passed; Ruff and mypy passed.

These targeted fixes do not rerun the protected scientific studies or update their recorded numerical findings.
