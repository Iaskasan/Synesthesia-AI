import importlib.util
from pathlib import Path


def load_download_module():
    module_path = Path(__file__).resolve().parents[1] / "mtg-jamendo-dataset" / "scripts" / "download" / "download.py"
    spec = importlib.util.spec_from_file_location("download_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_parser_accepts_hyphenated_dataset_and_outputdir_alias():
    module = load_download_module()
    parser = module.build_parser()

    args = parser.parse_args([
        "--dataset",
        "autotagging-moodtheme",
        "--type",
        "audio-low",
        "--from",
        "mtg-fast",
        "--outputdir",
        "/tmp/out",
        "--unpack",
        "--remove",
    ])

    assert args.dataset == "autotagging_moodtheme"
    assert args.type == "audio-low"
    assert args.download_from == "mtg-fast"
    assert args.outputdir == "/tmp/out"
    assert args.unpack is True
    assert args.remove is True


def test_download_from_mtg_resumes_from_partial_download(tmp_path, monkeypatch):
    module = load_download_module()
    output = tmp_path / "archive.tar"
    part_file = output.with_suffix(output.suffix + ".part")
    part_file.write_bytes(b"abc")

    class FakeResponse:
        def __init__(self):
            self.status_code = 206
            self.headers = {"Content-Length": "3"}

        def iter_content(self, chunk_size=512 * 1024):
            yield b"def"

    def fake_get(url, stream=True, headers=None, timeout=None):
        assert headers == {"Range": "bytes=3-"}
        return FakeResponse()

    monkeypatch.setattr(module.requests, "get", fake_get)

    module.download_from_mtg("https://example.com/archive.tar", str(output))

    assert output.read_bytes() == b"abcdef"
    assert not part_file.exists()
