"""One-shot Colab runner for the v4 bridge hazard experiment.

Downloads the pinned public bundles, rebuilds v3 and v4 from the same raw
files, runs the existing v3 benchmark (tabular models) and the v4 hazard
models on the same test windows, and prints one JSON summary. Raw data and
derived datasets stay in --work and are never committed.

In a Colab cell:

    !git clone -b <branch> https://github.com/Ayberkrk/cauren.git && cd cauren \\
        && pip -q install -e ".[dataset,benchmark]" pyarrow lightgbm \\
        && python tools/colab/run_bridge_v4.py --work /content/work
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HF_REVISIONS = {
    "bridgedeck-nbi": "4a26790d431ae515287bbd934bb3d8e7281f3a81",
    "bridgedeck-nshm": "9c514df711f47e8949b3fcaf883fd4851ecb3e2b",
    "bridgedeck-nfhl": "cd520c4f275b0cb534fc824a4cb86242826f6c34",
}
HF_URL = "https://huggingface.co/datasets/sweetapricity/{dataset}/resolve/{revision}/{path}"


def fetch(dataset: str, path: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = HF_URL.format(dataset=dataset, revision=HF_REVISIONS[dataset], path=path)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, target)
    return target


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path("/content/work"))
    parser.add_argument("--states", nargs="+", default=["CA", "IA", "PA"])
    parser.add_argument("--skip-v3-benchmark", action="store_true")
    args = parser.parse_args()

    raw = args.work / "nbi_raw"
    for state in args.states:
        archive = fetch("bridgedeck-nbi", f"data/{state}.zip", args.work / "zips" / f"{state}.zip")
        if not any(raw.glob(f"{state}[0-9][0-9].txt")):
            raw.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(raw)
    nshm = fetch("bridgedeck-nshm", "data/raw/nshm_2014_conus_0p05deg.csv", args.work / "nshm.csv")
    nfhl = fetch("bridgedeck-nfhl", "data/raw/nfhl_bridge_zones.csv", args.work / "nfhl.csv")

    v3 = args.work / "cauren_bridge"
    v4 = args.work / "cauren_bridge_v4"
    common = ["--raw-dir", str(raw), "--nshm-csv", str(nshm), "--nfhl-csv", str(nfhl), "--states", *args.states]
    if not (v3 / "dataset_summary.json").exists():
        sh("tools/build_cauren_bridge_dataset.py", *common, "--output-dir", str(v3))
    predictions = v3 / "model_predictions.csv"
    if not args.skip_v3_benchmark and not predictions.exists():
        sh("tools/benchmark_cauren_bridge_models.py", "--dataset-dir", str(v3), "--skip-hybrid",
           "--output-json", str(v3 / "model_benchmark.json"), "--output-predictions", str(predictions))
    if not (v4 / "bridge_years.parquet").exists():
        sh("tools/build_bridge_years_v4.py", *common,
           "--v3-windows", str(v3 / "agents" / "cauren-bridge" / "windows.csv"), "--output-dir", str(v4))
    hazard_args = ["tools/bridge_hazard.py", "--panel", str(v4 / "bridge_years.parquet"),
                   "--v3-dataset-dir", str(v3), "--output-dir", str(v4 / "hazard")]
    if predictions.exists():
        hazard_args += ["--compare-with", str(predictions)]
    sh(*hazard_args)
    calibration = v4 / "hazard" / "calibration.json"
    sh("tools/bridge_calibration.py", "--predictions", str(v4 / "hazard" / "hazard_predictions.csv"),
       "--model", "v4_hazard_end2023", "--output-json", str(calibration))

    report = {
        "v3_summary": json.loads((v3 / "dataset_summary.json").read_text()),
        "v4_summary": json.loads((v4 / "dataset_summary.json").read_text()),
        "hazard": json.loads((v4 / "hazard" / "hazard_results.json").read_text()),
        "calibration": json.loads(calibration.read_text()),
    }
    if (v3 / "model_benchmark.json").exists():
        benchmark = json.loads((v3 / "model_benchmark.json").read_text())
        report["v3_benchmark"] = {key: benchmark[key] for key in ("models", "leave_one_state_out_tabular_models") if key in benchmark}
    for summary in (report["v3_summary"], report["v4_summary"]):
        summary.pop("build_provenance", None)
    (args.work / "bridge_v4_report.json").write_text(json.dumps(report, indent=2))
    print("BRIDGE_V4_REPORT_BEGIN")
    print(json.dumps(report))
    print("BRIDGE_V4_REPORT_END")


if __name__ == "__main__":
    main()
