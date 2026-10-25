from pathlib import Path

from tools.provenance import file_record, package_versions, sha256_file


def test_sha256_file_streams_and_records_input_identity(tmp_path: Path):
    source = tmp_path / "public-source.csv"
    source.write_bytes(b"header,value\nrow,1\n")

    record = file_record(source, source_id="test-source")

    assert record["source_id"] == "test-source"
    assert record["file_name"] == "public-source.csv"
    assert record["size_bytes"] == source.stat().st_size
    assert record["sha256"] == "5525b1fb2c0cf101d019564687aaf0b224bc20a33506065096c165f7e8058674"
    assert sha256_file(source) == record["sha256"]


def test_package_versions_reports_unavailable_packages_explicitly():
    versions = package_versions(("pip", "cauren-package-that-does-not-exist"))

    assert versions["pip"]
    assert versions["cauren-package-that-does-not-exist"] is None
