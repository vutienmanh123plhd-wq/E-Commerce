import numpy as np

from app.data_access.vector_index import VectorIndex


def _unit(i, dim=512):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _index():
    ids = [10, 20, 30]
    img = np.stack([_unit(0), _unit(1), _unit(2)])
    txt = np.stack([_unit(3), _unit(4), _unit(5)])
    return VectorIndex(ids, img, txt)


def test_nearest_returns_matching_product_first():
    q = _unit(1) + 0.1 * _unit(0)
    res = _index().nearest(q, k=3, min_score=0.0, space="image")
    assert res[0][0] == 20
    assert res[0][1] > res[1][1]


def test_min_score_filters_unrelated():
    res = _index().nearest(_unit(1), k=3, min_score=0.5, space="image")
    assert [pid for pid, _ in res] == [20]


def test_k_limits_results():
    q = np.ones(512, dtype=np.float32)
    assert len(_index().nearest(q, k=1, min_score=-1.0, space="image")) == 1


def test_both_space_uses_text_vectors_too():
    res = _index().nearest(_unit(5), k=3, min_score=0.5, space="both")
    assert [pid for pid, _ in res] == [30]


def _big_index(n=20):
    rng = np.random.default_rng(0)
    img = rng.normal(size=(n, 512)).astype(np.float32)
    txt = rng.normal(size=(n, 512)).astype(np.float32)
    return VectorIndex(list(range(1, n + 1)), img, txt), img, txt


def test_fused_nearest_ranks_product_matching_both_spaces_first():
    idx, img, txt = _big_index()
    q = img[4] / np.linalg.norm(img[4]) + txt[4] / np.linalg.norm(txt[4])
    res = idx.fused_nearest(q[None, :], k=3, min_z=1.0)
    assert res[0][0] == 5


def test_fused_nearest_threshold_drops_unrelated_query():
    idx, _, _ = _big_index()
    q = np.random.default_rng(99).normal(size=(1, 512)).astype(np.float32)
    assert idx.fused_nearest(q, k=5, min_z=3.0) == []


def test_fused_nearest_takes_best_of_query_variants():
    idx, img, txt = _big_index()
    noise = np.random.default_rng(5).normal(size=512).astype(np.float32)
    good = img[7] + txt[7]
    res = idx.fused_nearest(np.stack([noise, good]), k=1, min_z=1.0)
    assert res[0][0] == 8


def test_fused_nearest_constant_scores_do_not_crash():
    idx = VectorIndex([1, 2], np.stack([_unit(0), _unit(0)]), np.stack([_unit(1), _unit(1)]))
    assert idx.fused_nearest(_unit(0)[None, :], k=2, min_z=0.5) == []


def test_empty_index_returns_nothing(tmp_path):
    idx = VectorIndex.load(tmp_path / "missing.npz")
    assert idx.is_empty
    assert idx.nearest(_unit(0), k=5, min_score=0.0, space="image") == []
    assert idx.fused_nearest(_unit(0)[None, :], k=5, min_z=0.0) == []


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "v.npz"
    _index().save(path)
    loaded = VectorIndex.load(path)
    assert loaded.product_ids == [10, 20, 30]
    assert loaded.nearest(_unit(2), k=1, min_score=0.5, space="image")[0][0] == 30
