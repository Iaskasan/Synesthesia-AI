"""Compare space-only heads on cached CLAP features; never load test embeddings.

Run: python -m src.ml.compare_space_heads
Models and results are separate from the application's multilabel checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.run_clap_diagnostics import validation_thresholds
from src.ml.train_space_candidate import binary_metrics


def metrics(truth, scores, threshold):
    return dict(**binary_metrics(truth, scores >= threshold),
                average_precision=float(average_precision_score(truth, scores)),
                roc_auc=float(roc_auc_score(truth, scores)))


def main():
    output = Path('artifacts/space_head_comparison')
    output.mkdir(parents=True, exist_ok=False)
    embedding_root = Path('/mnt/g/AI/datasets/processed/clap-music-v1')
    current = Path('artifacts/clap_space_candidate')
    review_paths = {
        'random_review': Path('artifacts/clap_diagnostics/validation_review_queue_calibration_check.csv'),
        'old_model_positive_review': current / 'space_positive_review_queue.csv',
    }
    reviews = {}
    excluded = set()
    for path in Path('artifacts').rglob('*queue*.csv'):
        with path.open() as handle:
            excluded.update(row['track_id'] for row in csv.DictReader(handle))
    for name, path in review_paths.items():
        with path.open() as handle:
            rows = [row for row in csv.DictReader(handle) if row['label'] == 'space']
        assert all(row['split'] == 'validation' and row['verdict'] in
                   ('correct', 'incorrect', 'ambiguous') for row in rows)
        reviews[name] = [row for row in rows if row['verdict'] != 'ambiguous']
    with (embedding_root / 'manifest.csv').open() as handle:
        manifest = list(csv.DictReader(handle))
    arrays = {}
    for split in ('train', 'validation'):
        rows = [row for row in manifest if row['split'] == split]
        if split == 'train':
            assert not excluded.intersection(row['track_id'] for row in rows)
        print(f'Loading {len(rows)} {split} embedding files...', flush=True)
        pooled, crops = [], []
        for row in rows:
            with np.load(embedding_root / row['embedding_path']) as archive:
                pooled.append(archive['embedding'])
                crops.append(archive['crop_embeddings'])
        arrays[split] = dict(pooled=np.asarray(pooled, dtype=np.float32),
                             crops=np.asarray(crops, dtype=np.float32),
                             truth=np.array(['space' in json.loads(row['tags']) for row in rows]),
                             ids=[row['track_id'] for row in rows])
        assert np.isfinite(arrays[split]['pooled']).all() and np.isfinite(arrays[split]['crops']).all()
    train, val = arrays['train'], arrays['validation']
    index = {track: i for i, track in enumerate(val['ids'])}
    calibration = np.array([track not in excluded for track in val['ids']])
    report = dict(test_evaluated=False, deployed=False,
                  training_tracks=len(train['ids']), training_positives=int(train['truth'].sum()),
                  threshold_calibration_tracks=int(calibration.sum()),
                  review_hashes={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in review_paths.items()},
                  interpretation='Exploratory model comparison. Models fit training data only. Alternative thresholds maximize dataset F1 on validation tracks excluding all review queues. Random human review and old-model-positive review are reported separately, never pooled. The latter cannot estimate population recall or new-model precision. Selecting models after inspecting human metrics requires fresh validation. Saved heads are space-only and are not drop-in application checkpoints.',
                  experiments={})
    predictions = {}

    def record(name, scores, threshold):
        result = dict(threshold=float(threshold), validation_positive_count=int((scores >= threshold).sum()),
                      dataset_calibration=metrics(val['truth'][calibration], scores[calibration], threshold))
        for review_name, rows in reviews.items():
            ids = [index[row['track_id']] for row in rows]
            truth = np.array([(row['predicted'] == 'true') == (row['verdict'] == 'correct') for row in rows])
            result[review_name] = metrics(truth, scores[ids], threshold)
        report['experiments'][name] = result
        predictions[name] = scores
        (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{name}: dataset AP={result['dataset_calibration']['average_precision']:.3f}; "
              f"random-review P/R={result['random_review']['precision']:.2f}/{result['random_review']['recall']:.2f}; "
              f"old-positive-review P/R={result['old_model_positive_review']['precision']:.2f}/{result['old_model_positive_review']['recall']:.2f}", flush=True)

    baseline = joblib.load(current / 'selected_head.joblib')
    with np.load(current / 'validation_predictions.npz') as cache:
        assert cache['track_ids'].tolist() == val['ids']
        j = cache['labels'].tolist().index('space')
        record('current_crop_c_0.01', cache['probabilities'][:, j], baseline['thresholds'][j])
    experiments = [(f'pooled_c_{c:g}', 'pooled', c, None) for c in (.001, .01, .1, 1.)]
    experiments += [(f'crop_c_{c:g}', 'crops', c, None) for c in (.001, .1)]
    experiments += [(f'pooled_rbf_gamma_{gamma:g}', 'pooled', .1, gamma)
                    for gamma in (1 / 512, 1 / 128)]
    for name, representation, c, gamma in experiments:
        print(f'Fitting {name}...', flush=True)
        steps = [StandardScaler()]
        if gamma is not None:
            steps.append(Nystroem(kernel='rbf', gamma=gamma, n_components=256, random_state=42))
        steps.append(LogisticRegression(C=c, class_weight='balanced', solver='liblinear',
                                        max_iter=2000, random_state=42))
        model = make_pipeline(*steps)
        x, vx, y = train[representation], val[representation], train['truth']
        if representation == 'crops':
            x = x.reshape(-1, x.shape[-1])
            y = np.repeat(y, train['crops'].shape[1])
            vx = vx.reshape(-1, vx.shape[-1])
        model.fit(x, y)
        assert model.named_steps['logisticregression'].n_iter_.max() < 2000
        scores = model.predict_proba(vx)[:, 1]
        if representation == 'crops':
            scores = scores.reshape(len(val['ids']), val['crops'].shape[1]).mean(axis=1)
        threshold = validation_thresholds(val['truth'][calibration, None], scores[calibration, None])[0]
        joblib.dump(dict(model=model, label='space', threshold=float(threshold), representation=representation,
                         crop_aggregation='mean' if representation == 'crops' else None), output / f'{name}.joblib')
        record(name, scores, threshold)
    np.savez_compressed(output / 'validation_predictions.npz', **predictions,
                        track_ids=np.array(val['ids']), truth=val['truth'], calibration_mask=calibration)
    print(f'Completed comparison: {output}. Live model unchanged.', flush=True)


if __name__ == '__main__':
    main()
