import numpy as np

from observables.stage12 import galleries, pair_accuracy, tasks


def test_hierarchical_galleries_are_balanced_and_source_disjoint():
    mapping = np.repeat(np.arange(20), 5)
    fine = np.repeat(np.arange(100), 10)
    data = dict(x_test=np.arange(len(fine))[:, None], fine=fine, mapping=mapping)
    selected, superclasses = [], []
    for _, _, ids, _, labels in galleries(data):
        assert len(ids) == 40
        assert np.array_equal(np.unique(labels, return_counts=True)[1], [2]*20)
        categories, counts = np.unique(mapping[labels], return_counts=True)
        assert np.array_equal(counts, [8]*5)
        superclasses.extend(categories)
        selected.extend(ids)
    assert len(selected) == len(set(selected)) == 480
    assert np.array_equal(np.bincount(superclasses), [3]*20)
    assert len(np.unique(fine[selected])) == 100


def test_hierarchical_targets_exclude_the_right_rows():
    ids = np.arange(40)
    fine = np.repeat(np.arange(20), 2)
    coarse = np.repeat(np.arange(5), 8)
    (name_fine, mask_fine, truth_fine), (name_coarse, mask_coarse, truth_coarse) = tasks(ids, 3, fine, coarse)
    assert (name_fine, name_coarse) == ('fine', 'coarse')
    np.testing.assert_array_equal(ids[mask_fine], [0, 1, 2, 4, 5, 6, 7])
    np.testing.assert_array_equal(ids[mask_fine][truth_fine], [2])
    np.testing.assert_array_equal(ids[mask_coarse], np.r_[0, 1, 4:40])
    np.testing.assert_array_equal(ids[mask_coarse][truth_coarse], [0, 1, 4, 5, 6, 7])
    assert pair_accuracy(np.zeros(38), truth_coarse) == .5
