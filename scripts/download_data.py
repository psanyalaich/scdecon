"""Idempotent downloader for the M7 real-data inputs, with provenance.

Dataset-specific orchestration (lives in ``scripts/``, not the package). Fetches
the TCGA-SKCM bulk (recount3), the GENCODE v26 annotation (recount3), and the
Tirosh melanoma scRNA reference (GEO GSE72056) into a local, git-ignored
``data/`` tree, recording a SHA-256 checksum, byte size, and retrieval time for
each file in a JSON manifest. Uses only the standard library (no new runtime
dependency); the data itself is never committed.

Re-running is idempotent: a source whose destination already exists is skipped
(its checksum is still recorded) unless ``--force`` is given. URLs/accessions were
verified live on 2026-07-10; the checksums pin the exact bytes retrieved.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --only tcga_gene_sums gencode_gtf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from scdecon.logging_utils import configure_logging, get_logger

logger = get_logger("scripts.download_data")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "raw"
_MANIFEST_NAME = "download_manifest.json"
_USER_AGENT = "scdecon-download/0.1 (+https://github.com/psanyalaich/scdecon)"
_CHUNK_SIZE = 1 << 20  # 1 MiB
_TIMEOUT_S = 120


@dataclass(frozen=True)
class DatasetSource:
    """A single downloadable input and where it should land.

    Attributes
    ----------
    name:
        Short, stable identifier (used for ``--only`` and in the manifest).
    url:
        Fully-resolved HTTPS download URL (verified live).
    filename:
        Destination file name within the data directory.
    description:
        Human-readable description of the file.
    accession:
        Source accession / project identifier for provenance.
    """

    name: str
    url: str
    filename: str
    description: str
    accession: str


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of fetching (or verifying) one source."""

    name: str
    url: str
    filename: str
    sha256: str
    n_bytes: int
    skipped: bool
    retrieved_at: str


#: The M7 inputs. Dataset-specific by design (kept out of the package).
SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="tcga_gene_sums",
        url=(
            "https://duffel.rail.bio/recount3/human/data_sources/tcga/gene_sums/"
            "CM/SKCM/tcga.gene_sums.SKCM.G026.gz"
        ),
        filename="tcga.gene_sums.SKCM.G026.gz",
        description="TCGA-SKCM bulk gene-level coverage counts (recount3, GENCODE v26)",
        accession="recount3:tcga/SKCM",
    ),
    DatasetSource(
        name="tcga_metadata",
        url=(
            "https://duffel.rail.bio/recount3/human/data_sources/tcga/metadata/"
            "CM/SKCM/tcga.tcga.SKCM.MD.gz"
        ),
        filename="tcga.tcga.SKCM.MD.gz",
        description="TCGA-SKCM sample metadata (recount3).",
        accession="recount3:tcga/SKCM",
    ),
    DatasetSource(
        name="gencode_gtf",
        url=(
            "https://duffel.rail.bio/recount3/human/annotations/gene_sums/"
            "human.gene_sums.G026.gtf.gz"
        ),
        filename="human.gene_sums.G026.gtf.gz",
        description="GENCODE v26 gene annotation (Ensembl-ID -> symbol bridge).",
        accession="GENCODE:v26/G026",
    ),
    DatasetSource(
        name="tirosh_scrna",
        url=(
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE72056&format=file"
            "&file=GSE72056%5Fmelanoma%5Fsingle%5Fcell%5Frevised%5Fv2%2Etxt%2Egz"
        ),
        filename="GSE72056_melanoma_single_cell_revised_v2.txt.gz",
        description="Tirosh et al. 2016 melanoma scRNA reference matrix.",
        accession="GEO:GSE72056",
    ),
)


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _sha256_and_size(path: Path, *, chunk_size: int = _CHUNK_SIZE) -> tuple[str, int]:
    """Return the SHA-256 hex digest and byte size of an existing file."""
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
            total += len(chunk)
    return hasher.hexdigest(), total


def _download_to(
    url: str,
    dest: Path,
    *,
    chunk_size: int = _CHUNK_SIZE,
    timeout: int = _TIMEOUT_S,
) -> tuple[str, int]:
    """Stream ``url`` to ``dest`` atomically, returning its SHA-256 and size.

    The body is written to a temporary file in the destination directory and
    :func:`os.replace`-d into place only on success, so an interrupted download
    never leaves a partial file at ``dest``. The connection is opened before the
    temporary file is created, so a failed request leaves no temp file behind;
    the temp file's descriptor is always closed before any cleanup unlink (which
    matters on Windows, where open files cannot be removed).

    When the server reports a ``Content-Length``, the number of bytes actually
    received is verified against it and an :class:`OSError` is raised on mismatch,
    guarding against silently truncated downloads. When the header is absent, no
    such check is performed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    hasher = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
        file_descriptor, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(file_descriptor, "wb") as out:
                while chunk := response.read(chunk_size):
                    out.write(chunk)
                    hasher.update(chunk)
                    total += len(chunk)
            if content_length is not None and content_length.isdigit():
                expected = int(content_length)
                if total != expected:
                    raise OSError(
                        f"Incomplete download from {url}: expected {expected} "
                        f"bytes but received {total}."
                    )
            os.replace(tmp_path, dest)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return hasher.hexdigest(), total


def download_source(
    source: DatasetSource,
    data_dir: Path,
    *,
    force: bool = False,
    chunk_size: int = _CHUNK_SIZE,
    timeout: int = _TIMEOUT_S,
) -> DownloadResult:
    """Download (or verify) one source into ``data_dir``.

    Parameters
    ----------
    source:
        The source to fetch.
    data_dir:
        Directory the file lands in (created if missing).
    force:
        If ``True``, re-download even when the destination already exists.
    chunk_size, timeout:
        Streaming chunk size (bytes) and socket timeout (seconds).

    Returns
    -------
    DownloadResult
        Provenance for the file, including its SHA-256 and whether it was skipped.
    """
    dest = data_dir / source.filename
    if dest.exists() and not force:
        logger.info("Skipping %s (already present): %s", source.name, dest.name)
        sha256, n_bytes = _sha256_and_size(dest, chunk_size=chunk_size)
        skipped = True
    else:
        logger.info("Downloading %s -> %s", source.name, dest)
        sha256, n_bytes = _download_to(
            source.url, dest, chunk_size=chunk_size, timeout=timeout
        )
        skipped = False
    return DownloadResult(
        name=source.name,
        url=source.url,
        filename=source.filename,
        sha256=sha256,
        n_bytes=n_bytes,
        skipped=skipped,
        retrieved_at=_utcnow(),
    )


def write_manifest(results: Sequence[DownloadResult], path: Path) -> Path:
    """Write a JSON provenance manifest for the downloaded sources.

    Parameters
    ----------
    results:
        The download results to record.
    path:
        Destination manifest path (parent directories are created).

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _utcnow(),
        "tool": "scripts/download_data.py",
        "note": "Provenance for M7 real-data inputs; data itself is never committed.",
        "sources": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: download the configured sources and write a manifest."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"Directory to download into (default: {_DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[source.name for source in SOURCES],
        help="Restrict to these source names (default: all).",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    force: bool = args.force
    only: list[str] | None = args.only

    selected = [source for source in SOURCES if not only or source.name in only]
    results = [download_source(source, data_dir, force=force) for source in selected]
    manifest = write_manifest(results, data_dir / _MANIFEST_NAME)
    logger.info("Wrote provenance manifest: %s", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
