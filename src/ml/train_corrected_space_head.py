"""Controlled space-label correction experiment; preserve threshold and other heads."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ml.run_clap_diagnostics import load_crop_split
from src.ml.train_space_candidate import binary_metrics


def make_overrides(reviews, training_tags):
    """Validate training-only dataset-reference reviews and reconstruct targets."""
    overrides = []
    seen = set()
    for row in reviews:
        track = row['track_id']
        if row['split'] != 'train' or row['label'] != 'space' or track not in training_tags:
            raise ValueError('Overrides must reference space reviews of training tracks only')
        if track in seen:
            raise ValueError('Duplicate training review')
        seen.add(track)
        target = int('space' in training_tags[track])
        reference = str(bool(target)).lower()
        if row['predicted'] != reference or row['dataset_target'] != reference:
            raise ValueError('Review reference differs from original dataset target')
        verdict = row['verdict']
        if verdict not in ('correct', 'incorrect', 'ambiguous'):
            raise ValueError('Every training review must have a valid verdict')
        new_target = None if verdict == 'ambiguous' else target if verdict == 'correct' else 1-target
        overrides.append(dict(track_id=track, split='train', label='space',
                              original_target=target, reviewed_target=new_target,
                              exclude_from_space_training=new_target is None, verdict=verdict,
                              notes=row.get('notes', '')))
    return overrides


def metrics(truth, scores, threshold):
    return dict(**binary_metrics(truth, scores >= threshold),
                average_precision=float(average_precision_score(truth, scores)),
                roc_auc=float(roc_auc_score(truth, scores)))


def main():
    source = Path('artifacts/clap_space_candidate')
    output = Path('artifacts/clap_space_corrected_candidate')
    output.mkdir(parents=True, exist_ok=False)
    embedding_root = Path('/mnt/g/AI/datasets/processed/clap-music-v1')
    manifest_path = embedding_root / 'manifest.csv'
    review_path = source / 'space_training_review_queue.csv'
    bundle_path = source / 'selected_head.joblib'
    live_path = Path('artifacts/clap_diagnostics/selected_head.joblib')
    watched = [manifest_path, review_path, bundle_path, live_path]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    with manifest_path.open() as handle:
        manifest = list(csv.DictReader(handle))
    train_rows = [r for r in manifest if r['split']=='train']
    val_rows = [r for r in manifest if r['split']=='validation']
    with review_path.open() as handle:
        reviews = list(csv.DictReader(handle))
    overrides = make_overrides(reviews, {r['track_id']: json.loads(r['tags']) for r in train_rows})
    assert len(overrides)==100
    (output/'training_overrides.json').write_text(json.dumps(overrides,indent=2)+'\n')
    mapping = {r['track_id']:r for r in overrides}
    keep = np.array([not mapping.get(r['track_id'],{}).get('exclude_from_space_training',False) for r in train_rows])
    print('Loading cached training crops...',flush=True)
    train_x, train_y = load_crop_split(manifest_path, embedding_root, 'train', ['space'])
    y = train_y[:,0].copy()
    for i,row in enumerate(train_rows):
        if row['track_id'] in mapping and keep[i]:
            y[i] = mapping[row['track_id']]['reviewed_target']
    assert int((~keep).sum())==9 and int(((y!=train_y[:,0]) & keep).sum())==28
    bundle = joblib.load(bundle_path)
    assert bundle['crop_aggregation']=='mean'
    j = bundle['labels'].index('space')
    threshold = float(bundle['thresholds'][j])
    model = clone(bundle['model'].estimators_[j])
    print(f'Fitting corrected space head on {int(keep.sum())} tracks; threshold stays {threshold:.6f}...',flush=True)
    x = train_x[keep]
    model.fit(x.reshape(-1,x.shape[-1]), np.repeat(y[keep],x.shape[1]))
    assert model.classes_.tolist()==[0,1]
    assert model.named_steps['logisticregression'].n_iter_.max()<2000
    print('Loading validation crops and comparing predictions...',flush=True)
    val_x, val_y = load_crop_split(manifest_path, embedding_root, 'validation', ['space'])
    scores = model.predict_proba(val_x.reshape(-1,val_x.shape[-1]))[:,1].reshape(len(val_x),val_x.shape[1]).mean(axis=1)
    with np.load(source/'validation_predictions.npz') as archive:
        assert archive['track_ids'].tolist()==[r['track_id'] for r in val_rows]
        old = archive['probabilities'][:,j].copy()
    indices = {r['track_id']:i for i,r in enumerate(val_rows)}
    report = dict(threshold=threshold, threshold_retuned=False, training_tracks=int(keep.sum()),
                  original_training_positives=int(train_y.sum()), corrected_training_positives=int(y[keep].sum()),
                  corrections=28, excluded_ambiguous=9, source_hashes=hashes,
                  experiment='Same cloned space estimator, hyperparameters, crop aggregation and threshold; only training targets and ambiguity exclusion change. Balanced class weights and scaler are refit as part of the same pipeline.',
                  test_evaluated=False, deployed=False, manual_reviews={},
                  dataset_validation={'before':metrics(val_y[:,0],old,threshold),'after':metrics(val_y[:,0],scores,threshold)},
                  limitation='Previously inspected validation samples are exploratory evidence. Old-model-positive review is selection-biased and cannot estimate new-model population precision or recall. Fresh validation is needed before promotion.')
    prediction_rows=[]
    for name,path in [('random_review',Path('artifacts/clap_diagnostics/validation_review_queue_calibration_check.csv')),
                      ('old_model_positive_review',source/'space_positive_review_queue.csv')]:
        with path.open() as handle:
            rr=[r for r in csv.DictReader(handle) if r['label']=='space' and r['verdict']!='ambiguous']
        assert all(r['split']=='validation' and r['verdict'] in ('correct','incorrect') for r in rr)
        assert not set(mapping).intersection(r['track_id'] for r in rr)
        ids=[indices[r['track_id']] for r in rr]
        truth=np.array([(r['predicted']=='true')==(r['verdict']=='correct') for r in rr])
        report['manual_reviews'][name]={'before':metrics(truth,old[ids],threshold),'after':metrics(truth,scores[ids],threshold)}
        for r,t,i in zip(rr,truth,ids):
            prediction_rows.append(dict(review=name,track_id=r['track_id'],human_target=int(t),before_probability=float(old[i]),after_probability=float(scores[i]),threshold=threshold,before_predicted=bool(old[i]>=threshold),after_predicted=bool(scores[i]>=threshold)))
    bundle['model'].estimators_[j]=model
    bundle['experiment']='crop_logistic_mean_space_training_corrections'
    joblib.dump(bundle,output/'selected_head.joblib')
    np.savez_compressed(output/'space_validation_predictions.npz',before=old,after=scores,track_ids=np.array([r['track_id'] for r in val_rows]))
    with (output/'review_predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in watched)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['manual_reviews'],indent=2),flush=True)
    print(f'Saved separate candidate to {output}; originals unchanged.',flush=True)


if __name__=='__main__':
    main()
