from src.data.load_data import load_dataset

def test_load_dataset(fake_dataset):
    ids, X, y = load_dataset(fake_dataset)
    assert len(ids) == 4
    assert X.shape == (4, 2)
    assert len(y) == 4