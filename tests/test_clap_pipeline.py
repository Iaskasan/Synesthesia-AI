import csv
import json

import numpy as np
import soundfile as sf

from src.ml.extract_clap_embeddings import (
    ClapEmbeddingConfig,
    extract_embeddings,
    fixed_crops,
    pool_crop_embeddings,
)
from src.ml.train_clap_head import convergence_report, load_split, parse_tags


def test_fixed_crops_are_deterministic_and_pad_short_audio():
    audio = np.arange(20, dtype=np.float32)
    crops = fixed_crops(audio, sample_rate=2, excerpt_seconds=12, crop_seconds=4)
    assert crops.shape == (3, 8)
    np.testing.assert_array_equal(crops[0], np.arange(8))
    np.testing.assert_array_equal(crops[-1], np.r_[np.arange(16, 20), np.zeros(4)])


def test_pool_crop_embeddings_normalizes_mean():
    pooled = pool_crop_embeddings(np.array([[3, 0], [0, 4]], dtype=np.float32))
    assert pooled.dtype == np.float32
    assert np.linalg.norm(pooled) == np.float32(1.0)


def test_extract_embeddings_resumes_and_writes_trainable_manifest(tmp_path):
    dataset = tmp_path / "audio"
    dataset.mkdir()
    sf.write(dataset / "one.wav", np.zeros(24, dtype=np.float32), 2)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "track_id", "artist_id", "audio_path", "feature_path", "tags"
        ))
        writer.writeheader()
        writer.writerow({
            "split": "train", "track_id": "track_1", "artist_id": "artist_1",
            "audio_path": "one.wav", "feature_path": "features/train/00/track_1.npz",
            "tags": json.dumps(["calm"]),
        })
    output = tmp_path / "clap"
    config = ClapEmbeddingConfig("fake", 2, 12, 4)

    def encoder(crops):
        return np.tile(np.array([[1, 2, 3]], dtype=np.float32), (len(crops), 1))

    first = extract_embeddings(manifest, dataset, output, config, encoder=encoder)
    second = extract_embeddings(manifest, dataset, output, config, encoder=encoder)
    assert first["processed"] == 1
    assert second["already_cached"] == 1
    x, y = load_split(output / "manifest.csv", output, "train", ["calm", "dark"])
    assert x.shape == (1, 3)
    assert y.tolist() == [[1, 0]]


def test_extract_embeddings_can_upgrade_cache_with_crop_embeddings(tmp_path):
    dataset = tmp_path / "audio"
    dataset.mkdir()
    sf.write(dataset / "one.wav", np.zeros(24, dtype=np.float32), 2)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "track_id", "artist_id", "audio_path", "feature_path", "tags"
        ))
        writer.writeheader()
        writer.writerow({
            "split": "train", "track_id": "track_1", "artist_id": "artist_1",
            "audio_path": "one.wav", "feature_path": "features/train/00/track_1.npz",
            "tags": json.dumps(["calm"]),
        })
    output = tmp_path / "clap"
    config = ClapEmbeddingConfig("fake", 2, 12, 4)
    calls = []

    def encoder(crops):
        calls.append(len(crops))
        return np.eye(3, dtype=np.float32)

    extract_embeddings(manifest, dataset, output, config, encoder=encoder)
    upgraded = extract_embeddings(
        manifest, dataset, output, config, encoder=encoder,
        store_crop_embeddings=True,
    )
    assert upgraded["processed"] == 1
    assert calls == [3, 3]
    with np.load(output / "embeddings/train/00/track_1.npz") as archive:
        assert archive["crop_embeddings"].shape == (3, 3)


def test_parse_tags_rejects_non_list_json():
    try:
        parse_tags('"calm"')
    except ValueError as error:
        assert "JSON list" in str(error)
    else:
        raise AssertionError("Expected invalid tags to fail")


def test_convergence_report_identifies_labels_at_limit():
    class Logistic:
        def __init__(self, iterations):
            self.n_iter_ = np.array([iterations])

    class Estimator:
        def __init__(self, iterations):
            self.named_steps = {"logisticregression": Logistic(iterations)}

    class Model:
        estimators_ = [Estimator(12), Estimator(20)]

    report = convergence_report(Model(), ["calm", "dark"], 20)
    assert report["iterations_by_label"] == {"calm": 12, "dark": 20}
    assert report["labels_at_iteration_limit"] == ["dark"]
