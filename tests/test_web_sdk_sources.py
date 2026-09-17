"""Where ltx studio gets helmstudio's SDK.

The page loads helm-css, the runtime's browser client and the components from
whatever runs the studio, `helm dev` or helmstudio, at /helm/sdk/v1 through the
runtime SDK's proxy, and nothing from anywhere else. The server's
helm-runtime-sdk comes from PyPI, at the major helmstudio.yaml pins. Nothing
names a clone of helmstudio.
"""

from __future__ import annotations

import re
import tomllib

import yaml

from tests.web_studio import REPO_ROOT

STATIC = REPO_ROOT / "web" / "static"
HOST_SDK = "/helm/sdk/v1/"

# What loads a resource in the page: tags, module imports and stylesheet URLs.
REFERENCES = [
    re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)\s*=\s*["']([^"']+)"""),
    re.compile(r"""\bimport\s*\(\s*[`"']([^`"']+)[`"']"""),
    re.compile(r"""\b(?:from|import)\s+["']([^"']+)["']"""),
    re.compile(r"""url\(\s*["']?([^"')\s]+)"""),
]
# The SDK's files, by the names the host serves them under.
SDK_FILE = re.compile(r"helm(?:-tokens|-base|-layout|-components)?(?:\.min)?\.css|helm-(?:runtime|ui)\.js|IBMPlex")


def page_references() -> dict[str, list[str]]:
    """Every resource each page file loads, with ${HELM_SDK} in app.js spelled out."""
    sdk = re.search(r"""const HELM_SDK = ["']([^"']+)["']""", (STATIC / "app.js").read_text())
    assert sdk, "app.js no longer names HELM_SDK; update this test with where the page loads the SDK"
    found: dict[str, list[str]] = {}
    for path in sorted(STATIC.rglob("*")):
        if path.suffix not in {".html", ".js", ".css"}:
            continue
        text = path.read_text().replace("${HELM_SDK}", sdk.group(1))
        found[path.name] = [ref for pattern in REFERENCES for ref in pattern.findall(text)]
    return found


def test_the_page_loads_the_sdk_from_the_host():
    refs = page_references()
    assert f"{HOST_SDK}helm-tokens.css" in refs["index.html"]
    assert f"{HOST_SDK}helm-runtime.js" in refs["app.js"]
    assert f"{HOST_SDK}helm-ui.js" in refs["app.js"]


def test_the_page_loads_nothing_from_anywhere_else():
    for name, refs in page_references().items():
        for ref in refs:
            assert not re.match(r"(?:[a-z]+:)?//", ref), f"{name} loads {ref} from another site"
            assert "node_modules" not in ref and "@helmstudio/" not in ref, f"{name} loads {ref} from npm"
            if SDK_FILE.search(ref):
                assert ref.startswith(HOST_SDK), f"{name} loads the SDK's {ref} from somewhere other than {HOST_SDK}"


def test_the_runtime_sdk_comes_from_pypi_at_the_manifests_major():
    manifest = yaml.safe_load((REPO_ROOT / "helmstudio.yaml").read_text())
    pinned = re.fullmatch(r"\^(\d+)", manifest["sdk"]["runtime"])
    assert pinned, f"helmstudio.yaml pins the runtime SDK as {manifest['sdk']['runtime']!r}; want a caret major"
    major = int(pinned.group(1))

    dependencies = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    spec = next(d for d in dependencies if re.match(r"helm-runtime-sdk\b", d))
    assert re.search(rf"<\s*{major + 1}(?:\D|$)", spec), f"{spec} does not stop below major {major + 1}"

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked = next(p for p in lock["package"] if p["name"] == "helm-runtime-sdk")
    assert locked["source"] == {"registry": "https://pypi.org/simple"}, (
        f"uv.lock takes helm-runtime-sdk from {locked['source']}"
    )
    assert int(locked["version"].split(".")[0]) == major


def test_nothing_names_a_clone_of_helmstudio():
    files = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "web" / "run.sh",
        REPO_ROOT / "web" / "server.py",
        *sorted((REPO_ROOT / ".vscode").glob("*.json")),
    ]
    for path in files:
        text = path.read_text()
        for local in ("HELMSTUDIO_REPO", "HELMSTUDIO_SDKS", "helmstudio-sdks", "packages/helm-runtime-sdk"):
            assert local not in text, f"{path.relative_to(REPO_ROOT)} names {local}"
