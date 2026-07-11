"""Unit tests for scripts.download_data (no network; file:// URLs)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.download_data import (
    SOURCES,
    DatasetSource,
    DownloadResult,
    download_source,
    main,
    write_manifest,
)

_PAYLOAD = b"hello scdecon\nsecond line\n"


def _local_source(tmp_path: Path, body: bytes = _PAYLOAD) -> DatasetSource:
    src = tmp_path / "src.gz"
    src.write_bytes(body)
    return DatasetSource(
        name="local",
        url=src.as_uri(),
        filename="out.gz",
        description="local test source",
        accession="TEST:1",
    )


def test_download_source_fetches_and_hashes(tmp_path: Path) -> None:
    source = _local_source(tmp_path)
    data_dir = tmp_path / "data"
    result = download_source(source, data_dir)

    dest = data_dir / "out.gz"
    assert dest.exists()
    assert dest.read_bytes() == _PAYLOAD
    assert result.skipped is False
    assert result.n_bytes == len(_PAYLOAD)
    assert result.sha256 == hashlib.sha256(_PAYLOAD).hexdigest()
    assert result.filename == "out.gz"


def test_download_source_is_idempotent_without_network(tmp_path: Path) -> None:
    source = _local_source(tmp_path)
    data_dir = tmp_path / "data"
    first = download_source(source, data_dir)

    # Delete the source: a second call must NOT re-fetch; it skips using the
    # already-present destination and still records the checksum.
    Path(tmp_path / "src.gz").unlink()
    second = download_source(source, data_dir)

    assert second.skipped is True
    assert second.sha256 == first.sha256
    assert second.n_bytes == first.n_bytes


def test_force_redownloads(tmp_path: Path) -> None:
    source = _local_source(tmp_path)
    data_dir = tmp_path / "data"
    download_source(source, data_dir)

    # Change the source and force a re-download.
    (tmp_path / "src.gz").write_bytes(b"v2 contents")
    result = download_source(source, data_dir, force=True)

    assert result.skipped is False
    assert (data_dir / "out.gz").read_bytes() == b"v2 contents"
    assert result.sha256 == hashlib.sha256(b"v2 contents").hexdigest()


def test_no_partial_file_on_failure(tmp_path: Path) -> None:
    source = DatasetSource(
        name="missing",
        url=(tmp_path / "does_not_exist.gz").as_uri(),
        filename="out.gz",
        description="broken source",
        accession="TEST:missing",
    )
    data_dir = tmp_path / "data"
    with pytest.raises(Exception):  # noqa: B017 - urllib raises URLError/FileNotFound
        download_source(source, data_dir)
    assert not (data_dir / "out.gz").exists()
    # No leftover .part temp files.
    assert not list(data_dir.glob("*.part"))


class _FakeResponse:
    """Minimal stand-in for a urlopen response (context manager + read + headers)."""

    def __init__(self, body: bytes, content_length: str | None) -> None:
        self._body = body
        self._pos = 0
        self.headers: dict[str, str] = (
            {} if content_length is None else {"Content-Length": content_length}
        )

    def read(self, size: int) -> bytes:
        chunk = self._body[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, body: bytes, content_length: str | None
) -> None:
    def fake_urlopen(request: object, timeout: object = None) -> _FakeResponse:
        return _FakeResponse(body, content_length)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def _remote_source() -> DatasetSource:
    return DatasetSource(
        name="remote",
        url="https://example.invalid/x.gz",
        filename="out.gz",
        description="remote test source",
        accession="TEST:remote",
    )


def test_truncated_download_raises_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Server claims 100 bytes but only 7 arrive.
    _patch_urlopen(monkeypatch, b"partial", content_length="100")
    data_dir = tmp_path / "data"
    with pytest.raises(OSError, match="Incomplete download"):
        download_source(_remote_source(), data_dir)
    assert not (data_dir / "out.gz").exists()
    assert not list(data_dir.glob("*.part"))


def test_matching_content_length_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"exactly right"
    _patch_urlopen(monkeypatch, body, content_length=str(len(body)))
    result = download_source(_remote_source(), tmp_path / "data")
    assert (tmp_path / "data" / "out.gz").read_bytes() == body
    assert result.n_bytes == len(body)


def test_absent_content_length_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"no length header"
    _patch_urlopen(monkeypatch, body, content_length=None)
    result = download_source(_remote_source(), tmp_path / "data")
    assert result.n_bytes == len(body)
    assert result.skipped is False


def test_write_manifest_roundtrips(tmp_path: Path) -> None:
    result = DownloadResult(
        name="local",
        url="file:///x",
        filename="out.gz",
        sha256="deadbeef",
        n_bytes=42,
        skipped=False,
        retrieved_at="2026-07-10T00:00:00+00:00",
    )
    path = write_manifest([result], tmp_path / "m.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "generated_at" in payload
    assert payload["tool"] == "scripts/download_data.py"
    assert payload["sources"][0]["sha256"] == "deadbeef"
    assert payload["sources"][0]["filename"] == "out.gz"


def test_sources_registry_is_wellformed() -> None:
    assert SOURCES
    names = [source.name for source in SOURCES]
    assert len(names) == len(set(names))  # unique names
    for source in SOURCES:
        assert source.url.startswith("https://")
        assert source.filename
        assert source.accession


def test_main_downloads_selected_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_download(
        source: DatasetSource, data_dir: Path, *, force: bool = False
    ) -> DownloadResult:
        calls.append(source.name)
        return DownloadResult(
            name=source.name,
            url=source.url,
            filename=source.filename,
            sha256="0" * 64,
            n_bytes=1,
            skipped=False,
            retrieved_at="2026-07-10T00:00:00+00:00",
        )

    monkeypatch.setattr("scripts.download_data.download_source", fake_download)
    target = SOURCES[0].name
    rc = main(["--data-dir", str(tmp_path), "--only", target])

    assert rc == 0
    assert calls == [target]
    manifest = json.loads((tmp_path / "download_manifest.json").read_text("utf-8"))
    assert [entry["name"] for entry in manifest["sources"]] == [target]
