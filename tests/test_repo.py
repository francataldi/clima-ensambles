import re
import subprocess
from pathlib import Path

from clima.viewer import social_head

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".py", ".md", ".yaml", ".yml", ".toml", ".bat", ".txt"}


def tracked_text_files():
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    files = [ROOT / name for name in listed if name]
    if not files:
        files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    return [p for p in files if p.suffix in TEXT and p.exists()]


def test_batch_files_are_ascii_crlf_and_point_to_real_files():
    bats = sorted(ROOT.glob("*.bat")) + sorted((ROOT / "scripts").glob("*.bat"))
    assert {b.name for b in bats} >= {"entrenar_todo.bat", "medir_todo.bat", "mantener_despierta.bat", "buscar_python.bat"}
    for bat in bats:
        raw = bat.read_bytes()
        raw.decode("ascii")
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), bat.name
        text = raw.decode("ascii")
        assert text.startswith("@echo off")
        for ref in re.findall(r"(?:scripts|configs)\\[\w.]+", text):
            assert (ROOT / ref.replace("\\", "/")).exists(), f"{bat.name}: {ref}"
    for name in ("entrenar_todo.bat", "medir_todo.bat"):
        text = (ROOT / name).read_text(encoding="ascii")
        assert 'cd /d "%~dp0"' in text and "buscar_python.bat" in text and "-m pytest" in text
        titles = re.findall(r"^title (\d+)/(\d+) ", text, flags=re.M)
        assert [int(a) for a, _ in titles] == list(range(1, len(titles) + 1))
        assert {int(b) for _, b in titles} == {len(titles)}


def test_no_hardware_names_anywhere():
    banned = [re.compile(p, re.I) for p in ("r" + "tx", r"\b30" + r"60\b", "ry" + "zen", "ge" + "force")]
    for path in tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in banned:
            assert not pattern.search(text), f"{path.relative_to(ROOT)}: {pattern.pattern}"
            assert not pattern.search(path.name), path.name


def test_documents_point_to_real_files():
    for doc in ("README.md", "TECNICO.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for ref in set(re.findall(r"(?:scripts|configs|docs/img)/[\w./-]+\.(?:py|yaml|png|jpg|gif)", text)):
            assert (ROOT / ref).exists(), f"{doc}: {ref}"
        for ref in set(re.findall(r"`(\w+\.bat)`", text)):
            assert (ROOT / ref).exists(), f"{doc}: {ref}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "TECNICO.md" in readme and "workflows/tests.yml" in readme
    assert (ROOT / ".github" / "workflows" / "tests.yml").exists()


def test_social_head_only_when_there_is_something_to_share():
    assert social_head({"title": "Visor"}) == ""
    head = social_head({"title": 'El "clima"', "description": "Un visor", "page_url": "https://ejemplo.org/", "image_url": "https://ejemplo.org/img.jpg"})
    assert '<meta name="description" content="Un visor">' in head
    assert '<meta property="og:title" content="El &quot;clima&quot;">' in head
    assert '<meta property="og:image" content="https://ejemplo.org/img.jpg">' in head
    assert '<meta name="twitter:card" content="summary_large_image">' in head
    assert head.count("\n") == 8
