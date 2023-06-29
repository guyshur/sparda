# Sparda - a sparse dataframe-like data structure

(Not to be confused with the other [Sparda](https://devilmaycry.fandom.com/wiki/Sparda)).

Sparda is a module that defines SparseDataFrame (shortened to sdf), a data structure for working with sparse and big data. SparseDataFrame allows working with a large amount (in samples and/or features) of sparse data in a convenient way by defining methods for managing and filtering data with optional index identifiers and/or column names, much like a pandas dataframe. In addition, sample labels can be provided which enable functionalities such as stratified train/test splitting. Data is stored in a [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) matrix as the backend. 

## Installation

Currently sparda offers a single module of the same name. Simply clone the repository/download the file and add to your project.

## Examples

### Instantiating a SparseDataFrame
There are currently three ways to create a SparseDataFrame:

 - Calling the constructor directly with a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix:  
 ` sdf = sparda.SparseDataFrame(matrix, index, columns, labels)`
 - With a dense numpy array:  
 `sdf = sparda.from_dense_array(array, index, columns, labels)`
 - With a sparse dict of the shape:  
>{(sample_1,feature_1): value, (sample_1,feature_2): value, (sample_2,feature_2): value, (sample_2,feature_3): value, ...}
>
and label dict of the shape:  
> {sample_1: label_1, sample_2: label_2, sample_3: label_1, ... }
>
  
 `sdf = sparda.from_sparse_dict(sparse_dict, label_dict)`
### Updating a SparseDataFrame
Updating a SparseDataFrame can be done by calling the `update` method with a sparse dict of the shape similar to the one above, using the number of the row and column in the dictionary tuple keys instead of the row and column names. To use row and column names instead, use with `use_row_col_names` set to True. Note that to add new samples or features, use the 'concat_vertically' and 'concat_horizontally' methods respectively. 
### Data cleaning and sampling
dropping zero-vector columns:
`sdf = sdf.drop_zero_columns()`
taking a random sample of the data (not stratified):
`sdf = sdf.random_sample_frac(fraction=0.1)`
negative sampling:
`negative_sample_sdf = sdf.negative_sampling(label=label, n_samples_n_samples)`

### Stratified train/test split
    sdf_train, sdf_test = sdf.get_stratified_train_test_split(test_size=0.2)
Note that `get_stratified_train_test_split` will fail if any sample has a unique label, in which case these samples must be filtered:
`sdf = sdf.drop_singleton_labels()`
One may choose, when not every label is important or during early development, to filter all samples with labels below a certain count:
`sdf = sdf.drop_low_frequency_labels(min_counts)

### Custom masks
Selecting specific samples by index/label and columns by name can be done by creating a boolean iterable, conditioning on sdf.index_ids, sdf.labels or sdf.column_names:
    idx_mask = [True if sample_id.startswith('batch1') else False for sample_id in sdf.index_ids]
    batch1_sdf = sdf._keep_indices(mask)
    # house_keeping_genes = set(...)
    col_mask = [True if column_name in house_keeping_genes else False for column_name in batch1_sdf.column_names]
    batch1_hkg_sdf = batch1_sdf._keep_columns(mask)
Both:
    batch1_hkg_sdf = sdf._keep_indices_and_columns(idx_mask, col_mask)
### Machine learning usage
Many scikit-learn algorithms can work with scipy.sparse matrices directly:

Standard scaler (without mean):

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=False)
    sdf_train.matrix = scaler.fit_transform(sdf_train.matrix)
    sdf_test.matrix = scaler.transform(sdf_test.matrix)
MaxAbsScaler:

    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    sdf_train.matrix = scaler.fit_transform(sdf_train.matrix)
    sdf_test.matrix = scaler.transform(sdf_test.matrix)
Training a model:

    from xgboost import XGBClassifier
    clf = XGBClassifier()
    X = sdf_train.matrix
    y = sdf_train.labels
    clf.fit(X, y)
    X_test = sdf_test.matrix
    y_test = sdf_test.labels
    y_preds = clf.predict(X_test)
Tensors:

Getting a sparse tensorflow tensor:
`tensor = sdf.to_tensorflow()`
Getting a spasre pytorch tensor:
`tensor = sdf.to_torch()`
