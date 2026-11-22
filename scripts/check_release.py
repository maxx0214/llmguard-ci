"""Check release metadata and archive contents without extracting or uploading."""

import argparse
import hashlib
import re
import tarfile
import zipfile
from email import message_from_bytes
from pathlib import Path, PurePosixPath


PACKAGE = "llmguard-ci"
PUBLIC_FILES = {
    "README.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md", "MANIFEST.in",
    "pyproject.toml", "requirements.txt", "docs/usage.md", "docs/benchmarks.md",
    "docs/releasing.md", "docs/examples/comparison.html", "scripts/check_release.py",
    "config/evaluation.yaml", "config/pricing.yaml", "config/thresholds.yaml",
    "prompts/banking77_v1.yaml", "prompts/banking77_v2.yaml",
    "examples/quickstart.py", "examples/suite/baseline.jsonl", "examples/suite/candidate.jsonl",
    "examples/suite/retrieval.jsonl", "examples/suite/retrieval.yaml",
    "examples/suite/retrieval-candidate.yaml", "benchmarks/requirements.txt",
}
PACKAGE_FILES = re.compile(r"llmguard/(?:[A-Za-z_]\w*/)*[A-Za-z_]\w*\.py|llmguard/report/templates/report\.html")
SENSITIVE = re.compile(
    rb"(?<![\w-])(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"github_pat_[A-Za-z0-9_]{40,}|pypi-[A-Za-z0-9_-]{40,}|"
    rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|[A-Z]:[\\/]Users[\\/]|"
    rb"rag_" rb"testset_retriever_v1v2|coverageaware_" rb"v2)"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scan(name, content):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name, "Unsafe archive member path")
    require(SENSITIVE.search(content) is None, f"Potential sensitive material in {name}")


def check_metadata(raw, version):
    metadata = message_from_bytes(raw)
    require(metadata["Name"] == PACKAGE and metadata["Version"] == version, "Unexpected distribution name or version")
    require(metadata["License-Expression"] == "MIT" and "LICENSE" in metadata.get_all("License-File", []),
            "Missing MIT license metadata")
    require(all("extra ==" in dependency for dependency in metadata.get_all("Requires-Dist", [])),
            "The core package must have zero runtime dependencies")


def check_wheel(path, name, version):
    prefix = f"{name}.dist-info/"
    metadata_files = {"METADATA", "WHEEL", "RECORD", "top_level.txt", "entry_points.txt", "licenses/LICENSE"}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "Invalid wheel archive")
        for entry in names:
            require(PACKAGE_FILES.fullmatch(entry) or
                    (entry.startswith(prefix) and entry.removeprefix(prefix) in metadata_files),
                    f"Unexpected wheel member: {entry}")
            scan(entry, archive.read(entry))
        require({"llmguard/__init__.py", "llmguard/report/templates/report.html",
                 *(prefix + item for item in metadata_files)} <= set(names), "Missing wheel files")
        check_metadata(archive.read(prefix + "METADATA"), version)


def check_sdist(path, name, version):
    names = set()
    generated = {"PKG-INFO", "setup.cfg"}
    egg_info_files = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt"}
    with tarfile.open(path) as archive:
        for member in archive:
            require(member.name == name or member.name.startswith(f"{name}/"), "Invalid source archive root")
            entry = member.name.removeprefix(f"{name}/")
            if member.isdir():
                require(".." not in PurePosixPath(entry).parts and "\\" not in entry, "Unsafe archive directory")
                continue
            require(member.isfile() and entry not in names, f"Invalid archive member: {entry}")
            names.add(entry)
            require(entry in PUBLIC_FILES or entry in generated or
                    re.fullmatch(r"tests/(?:conftest|test_\w+)\.py|benchmarks/\w+\.py", entry) or
                    (entry.startswith("src/") and PACKAGE_FILES.fullmatch(entry.removeprefix("src/"))) or
                    (entry.startswith("src/llmguard_ci.egg-info/") and entry.rsplit("/", 1)[-1] in egg_info_files),
                    f"Unexpected source member: {entry}")
            raw = archive.extractfile(member).read()
            scan(entry, raw)
            if entry == "PKG-INFO":
                check_metadata(raw, version)
        missing = (PUBLIC_FILES | {"PKG-INFO", "src/llmguard/__init__.py"}) - names
        require(not missing, f"Missing source files: {', '.join(sorted(missing))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", default="0.3.0")
    args = parser.parse_args()
    name = f"llmguard_ci-{args.version}"
    wheel = args.directory / f"{name}-py3-none-any.whl"
    sdist = args.directory / f"{name}.tar.gz"
    require(set(args.directory.iterdir()) == {wheel, sdist}, "Expected only the target wheel and sdist")
    check_wheel(wheel, name, args.version)
    check_sdist(sdist, name, args.version)
    for path in (wheel, sdist):
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")


if __name__ == "__main__":
    main()
