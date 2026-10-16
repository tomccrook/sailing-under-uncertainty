# Public source release

This repository is a reviewed source snapshot of the completed local project. The original research checkout and its Git history are retained separately. The publication commits use their actual dates; they are not a reconstruction of daily work on the study.

The public package contains the simulator, controllers, training/evaluation implementations, tests, schemas, optional support extensions and local replay workbench. It excludes raw tracks, downloaded observations, training weights, run databases, cloud state, credentials and generated research directories.

The README and research documents distinguish findings recorded in the original studies from the quick-start example. Generated figure links in archived reports are local reproduction outputs; the live portfolio provides a curated recorded demonstration.

The release adds a small episode example and public-facing documentation. It does not retune controllers or rerun the protected scientific studies. The verification below applies to this release checkout, rather than relying on earlier recorded checks.

## Release verification — 16 October 2026

Core: 147 tests passed on CPython 3.12.15. Support extension: 43 passed. Research extension: 16 passed (206 total). The small episode example, Ruff and root mypy checks passed.

Dependency locks were installed into separate environments. The source-publication review checked the selected file list, private paths, credential patterns, excluded data/weight formats and historical defensive outputs. These checks do not rerun the full protected scientific studies. Upstream dependency deprecation warnings remain visible in test output.

The local frontend production build passed with Node 24.21.0. The GitHub Actions workflow includes Python checks and frontend compilation; hosted-run status is separate from these local results.
