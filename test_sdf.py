import sparda
import pytest
import numpy as np
import random
from scipy.sparse import csr_matrix


def generate_mock_sparse_numpy_array(dtype):
    random.seed(42)
    output = np.random.rand(100, 100)
    output[output < 0.9] = 0
    output*=100
    output = output.astype(dtype)
    return output

def generate_mock_sparse_dict_and_labels():
    random.seed(42)
    sparse_dict = {}
    for i in range(100):
        for j in range(100):
            if np.random.rand() > 0.9:
                sparse_dict[(str(i),str(j))] = np.random.rand()*100
    
    labels = np.random.randint(0, 10, 100)
    labels = {str(i):str(j) for i, j in enumerate(labels)}
    return sparse_dict, labels

def generate_data_for_sdf():
    array = generate_mock_sparse_numpy_array(np.float32)
    index_ids = [str(i) for i in range(100)]
    labels = random.choices([str(i) for i in range(10)], k=100)
    column_names = [str(i) for i in range(100)]
    return array, index_ids, labels, column_names

@pytest.fixture
def generate_mock_sdf():
    array, index_ids, labels, column_names = generate_data_for_sdf()
    return sparda.SparseDataFrame(
        matrix=csr_matrix(array),
        index_ids=index_ids,
        labels=labels,
        column_names=column_names,
        dtype=np.int16
    )

@pytest.mark.testable
def test_init():
    array, index_ids, labels, column_names = generate_data_for_sdf()
    csr = csr_matrix(array)
    sparda.SparseDataFrame(
        matrix=csr,
        index_ids=index_ids,
        labels=labels,
        column_names=column_names,
        dtype=np.int16
    )

    sparda.from_dense_array(
        data=array,
        index_ids=index_ids,
        labels=labels,
        column_names=column_names,
        dtype=np.int16
    )

    sparse_dict, labels = generate_mock_sparse_dict_and_labels()
    sparda.from_sparse_dict(
        data=sparse_dict,
        labels=labels,
        dtype=np.int16
    )
    sparda.from_sparse_dict(
        data=sparse_dict,
        dtype=np.int16
    )

    with pytest.raises(TypeError):
        sparda.from_sparse_dict(
            data=sparse_dict,
            labels=1,
            dtype=np.int16
        )


    with pytest.raises(TypeError):
        sparda.SparseDataFrame(
            matrix=array,
            index_ids=index_ids,
            labels=labels,
            column_names=column_names,
            dtype=np.int16
        )

    with pytest.raises(TypeError):
        sparda.SparseDataFrame(
            matrix=csr,
            index_ids=0,
            labels=labels,
            column_names=column_names,
            dtype=np.int16
        )
    
    with pytest.raises(TypeError):
        sparda.SparseDataFrame(
            matrix=csr,
            index_ids=index_ids,
            labels=0,
            column_names=column_names,
            dtype=np.int16
        )
    
    with pytest.raises(TypeError):
        sparda.SparseDataFrame(
            matrix=csr,
            index_ids=index_ids,
            labels=labels,
            column_names=0,
            dtype=np.int16
        )
    
    with pytest.raises(TypeError):
        sparda.SparseDataFrame(
            matrix=csr,
            index_ids=index_ids,
            labels=labels,
            column_names=column_names,
            dtype=0
        )

@pytest.mark.testable
def test_update(generate_mock_sdf):
    sdf:sparda.SparseDataFrame = generate_mock_sdf
    new_col_0_data = {
        (i,0): 0 for i in range(100)
    }
    sdf = sdf.update(new_col_0_data)
    sdf.matrix = sdf.matrix.tocsc()
    assert sdf.matrix.indptr[0] == sdf.matrix.indptr[1] == 0
    new_col_0_data = {
        (i,0): 1 for i in range(100)
    }
    sdf = sdf.update(new_col_0_data)
    sdf.matrix = sdf.matrix.tocsc()
    assert sdf.matrix.indptr[0] == 0
    assert sdf.matrix.indptr[1] == 100
    sdf.index_ids = np.array([str(i) for i in range(100)])
    sdf.column_names = np.array([str(i) for i in range(100)])
    new_col_0_data = {
        (str(i),'0'): 0 for i in range(100)
    }
    sdf = sdf.update(new_col_0_data, use_row_col_names=True)
    sdf.matrix = sdf.matrix.tocsc()
    assert sdf.matrix.indptr[0] == sdf.matrix.indptr[1] == 0
    new_col_0_data = {
        (str(i),'0'): 1 for i in range(100)
    }
    sdf = sdf.update(new_col_0_data, use_row_col_names=True)
    sdf.matrix = sdf.matrix.tocsc()
    assert sdf.matrix.indptr[0] == 0
    assert sdf.matrix.indptr[1] == 100
    new_col_0_data = {
        (str(i),'0'): 1.5 for i in range(100)
    }
    
    with pytest.warns(Warning):
        sdf = sdf.update(new_col_0_data, use_row_col_names=True)
    assert sdf.matrix[0,0] == 1
    with pytest.raises(TypeError):
        sdf = sdf.update(new_col_0_data, use_row_col_names=False)
    
@pytest.mark.testable
def test_train_test_split(generate_mock_sdf):

    sdf: sparda.SparseDataFrame = generate_mock_sdf

    # add a singleton label
    data = np.random.rand(1, sdf.matrix.shape[1])
    label = ['abcdefg']
    cols = sdf.column_names
    index = ['123456']
    single_sample_sdf = sparda.from_dense_array(data, index, label, cols, np.int16)
    combined_sdf = sdf.concat_vertically(single_sample_sdf)
    with pytest.raises(ValueError):
        sdf_train, sdf_test = combined_sdf.get_stratified_train_test_split(test_size=0.2)
    combined_sdf = combined_sdf.drop_singleton_labels()
    sdf_train, sdf_test = combined_sdf.get_stratified_train_test_split(test_size=0.2)

    assert set(sdf_train.index_ids).intersection(set(sdf_test.index_ids)) == set()

@pytest.mark.testable
def test_getitem(generate_mock_sdf):
    sdf: sparda.SparseDataFrame = generate_mock_sdf

    _ = sdf[0]

    _ = sdf[0:5]

    _ = sdf[[0,1,2]]

    _ = sdf[0:5, 0:5]

    for key in (
        'aaa', (slice(0,5), slice(0,5), slice(0,5)), 
        (slice(0,5), '0:5'), None):
        with pytest.raises(TypeError):
            _ = sdf[key]
    for key in ([1,2,1000000], [1,2,-1], -1, 1000000):
        with pytest.raises(ValueError):
            _ = sdf[key]
    

@pytest.mark.testable
def test_selection(generate_mock_sdf):

    sdf: sparda.SparseDataFrame = generate_mock_sdf

    random_label = random.choice(sdf.labels)
    _ = sdf.remove_label(random_label)

    _ = sdf.get_label_to_indices()
    
    _ = sdf.drop_labels_below_threshold(10000)

    _ = sdf.get_label_to_index_ids()

@pytest.mark.testable
def test_drop_zero_columns(generate_mock_sdf):
    sdf: sparda.SparseDataFrame = generate_mock_sdf
    assert sdf.dtype == np.int16
    sdf = sdf.update({(0,i):1 for i in range(sdf.matrix.shape[1])})
    assert sdf.dtype == np.int16
    assert sdf.matrix.shape == sdf.drop_zero_columns().matrix.shape
    before = sdf.matrix.shape
    assert sdf.dtype == np.int16
    sdf = sdf.update({(i,0):0 for i in range(sdf.matrix.shape[0])})
    sdf = sdf.drop_zero_columns()
    after = sdf.matrix.shape
    assert before[1] == after[1] + 1
