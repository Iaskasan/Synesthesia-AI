import pytest

from src.ml.train_corrected_space_head import make_overrides


def review(track, target, verdict, split='train'):
    return dict(track_id=track, label='space', split=split, predicted=target,
                dataset_target=target, verdict=verdict)


def test_training_overrides_reconstruct_targets_and_mask_ambiguity():
    rows = [review('a','true','incorrect'), review('b','false','incorrect'),
            review('c','true','ambiguous'), review('d','false','correct')]
    out = make_overrides(rows, {'a':['space'],'b':[],'c':['space'],'d':[]})
    assert [r['reviewed_target'] for r in out] == [0,1,None,0]
    assert [r['exclude_from_space_training'] for r in out] == [False,False,True,False]


@pytest.mark.parametrize('split', ['validation','test'])
def test_training_overrides_reject_other_splits(split):
    with pytest.raises(ValueError, match='training tracks only'):
        make_overrides([review('a','true','correct',split)], {'a':['space']})


def test_training_overrides_reject_mismatched_reference_and_duplicates():
    row = review('a','true','correct')
    with pytest.raises(ValueError, match='reference differs'):
        make_overrides([row], {'a':[]})
    with pytest.raises(ValueError, match='Duplicate'):
        make_overrides([row,row], {'a':['space']})
