"""
A class for working with sparse data for data science projects.
Although pandas dataframes can store sparse data in the form of sparse pandas series, operations on them are still slow.
In the common case of data science projects, they also usually still store labels for each row inside the dataframe.
The SparseDataFrame class is a wrapper around a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.
It can also store index identifiers in addition to positional indices and column names like a pandas dataframe, but in a seperate data structure. 
In addition it can strore labels for each row, and perform operations on the data based on the annotation of each row and label.
All operations are performed on the underlying matrix, so they are very fast on large and sparse matrices.
"""
from __future__ import annotations
import warnings
from types import UnionType
from typing import Iterable
import numpy as np
import random
from math import ceil
from dataclasses import dataclass
from scipy.sparse import csr_matrix, csc_matrix, hstack, vstack
import torch
import tensorflow as tf

def from_dense_array(
	data: np.ndarray,
	index_ids: Iterable | None = None,
	labels: Iterable | None = None,
	column_names: Iterable | None = None,
	dtype=np.float64,
	matrix_type=csc_matrix):
	"""
	Create a SparseDataFrame instance from a dense numpy array.
	:param data: The data to store in the dataframe.
	:param index_ids: Optional. The index ids (not necessarily numeric) of the rows in the dataframe.
	:param labels: Optional. The labels of the rows in the dataframe.
	:param column_names: Optional. The names of the columns in the dataframe.
	:param dtype: the dtype of the data. Defaults to np.float64.
	:param matrix_type: The type of the underlying matrix. Defaults to scipy.sparse.csc_matrix.
	"""
	if not isinstance(data, np.ndarray):
		raise TypeError('data must be a numpy array')
	if not matrix_type in [csr_matrix, csc_matrix]:
		raise TypeError('matrix_type must be either scipy.sparse.csr_matrix or scipy.sparse.csc_matrix')
	matrix = matrix_type(data.astype(dtype))
	return SparseDataFrame(matrix, index_ids, labels, column_names, dtype=dtype)

def from_sparse_dict(
	data: dict, 
	labels: dict | None = None,
	dtype=np.float64):
	"""
	Create a SparseDataFrame from a dict of (index_id, feature_name) -> value and a dict of index_id -> label.
	"""
	if not type(data) == dict:
		raise TypeError('data must be a dict')
	if labels is not None and not type(labels) == dict:
		raise TypeError('labels must be a dict or None')
	index_ids = list(set([x[0] for x in list(data.keys())])) # unordered list of unique ids
	features_to_indices = {} # where we will store the mapping from feature name to index of the feature in the matrix
	feature_index_counter = 0 # used to assign indices to features
	index_ids_to_indices = {index_id: i for i, index_id in enumerate(index_ids)} # maps index ids to indices in the matrix
	rows, cols, values = [], [], []
	for id_and_feature, value in data.items(): # input is map of (index_id, feature_name) -> value, goal is to unpack to list of index_id indices, list of feature indices and list of values
		feature = id_and_feature[1]
		if feature not in features_to_indices:
			features_to_indices[feature] = feature_index_counter # assign index to new feature
			feature_index_counter += 1
		# get indices of id and feature in the matrix, and append them to the lists
		rows.append(index_ids_to_indices[id_and_feature[0]])
		cols.append(features_to_indices[feature])
		values.append(value)
	values = np.array(values).astype(dtype)
	sm = csr_matrix((values, (rows, cols)))
	# get sorted index_ids by their index
	sorted_index_ids = [index_id for index_id in sorted(index_ids_to_indices, key=index_ids_to_indices.get)]
	# get sorted labels by their index
	if labels is not None:
		sorted_labels = [labels[index_id] for index_id in sorted_index_ids]
	else:
		sorted_labels = None
	# get sorted features by their index
	sorted_columns = [feature for feature in sorted(features_to_indices, key=features_to_indices.get)]
	return SparseDataFrame(sm, sorted_index_ids, sorted_labels, sorted_columns, dtype=dtype)

@dataclass
class SparseDataFrame(object):
	"""
	A class for storing sparse data with many rows and columns in a dataframe-like format
	and performing fast operations on it.
	"""
	matrix: csr_matrix | csc_matrix
	index_ids: Iterable | None = None
	labels: Iterable | None = None
	column_names: Iterable | None = None
	dtype: np.dtype = np.float64
	def __init__(self,
				matrix: csr_matrix | csc_matrix,
				index_ids: Iterable | None = None,
				labels: Iterable | None = None,
				column_names: Iterable | None = None,
				dtype=np.float64):
		
		if not type(matrix) == csr_matrix and not type(matrix) == csc_matrix:
			raise TypeError('matrix must be a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix')
		if column_names is not None:
			if not isinstance(column_names, Iterable):
				raise TypeError('column_names must be an iterable object such as a list, tuple or a numpy array')
			if not len(column_names) == matrix.shape[1]:
				raise ValueError('column_names must have the same length as the number of columns in the matrix')
		if index_ids is not None:
			if not isinstance(index_ids, Iterable):
				raise TypeError('index_ids must be an iterable object such as a list, tuple or a numpy array')
			if not len(index_ids) == matrix.shape[0]:
				raise ValueError('index_ids must have the same length as the number of rows in the matrix')
			if not len(index_ids) == len(set(index_ids)):
				raise ValueError('index_ids must be unique')
		if labels is not None:
			if not isinstance(labels, Iterable):
				raise TypeError('labels must be an iterable object such as a list, tuple or a numpy array')
			if not len(labels) == matrix.shape[0]:
				raise ValueError('labels must have the same length as the number of rows in the matrix')
			
		matrix = matrix.astype(dtype)
		self.dtype = dtype
		self.matrix:csr_matrix = matrix
		self.index_ids = np.array(index_ids) if index_ids is not None else None
		self.labels = np.array(labels) if labels is not None else None
		self.column_names = np.array(column_names) if column_names is not None else None
	
	def __len__(self):
		return len(self.index_ids)

	def __getitem__(self, key: UnionType[int, slice, Iterable, tuple[slice, slice]]):
		if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], slice):
			idx_slice, col_slice = key[0], key[1]
			return SparseDataFrame(self.matrix[idx_slice, col_slice], self.index_ids[idx_slice], self.labels[idx_slice], self.column_names[col_slice], self.dtype)
		elif isinstance(key, slice):
			return SparseDataFrame(self.matrix[key, :], self.index_ids[key], self.labels[key], self.column_names, self.dtype)
		elif isinstance(key, Iterable):
			if all([type(x) == bool for x in key]):
				return self._keep_indices(key)
			elif all([type(x) == int for x in key]):
				if not 0 <= np.min(key):
					raise ValueError("Negative index in multiple row selection.")
				if not np.max(key) < len(self):
					raise ValueError("out of range index in multiple row selection.")
				return self._keep_indices([True if i in set(key) else False for i in range(len(self))])
			else:
				raise TypeError("Iterable must be booleans or integers.")
		elif isinstance(key, int):
			if not 0 <= key <= len(self):
				raise ValueError('Integer key is negative or out of range.')
		else:
			raise TypeError('Invalid argument type')

	def _keep_indices(self, mask: Iterable):
		"""
		Returns a new SparseDataFrame that only contains the rows that are True in the mask.
		"""
		if not any(mask):
			warnings.warn('No indices selected.')
		return SparseDataFrame(self.matrix[mask, :], self.index_ids[mask], self.labels[mask], self.column_names, self.dtype)

	def _keep_columns(self, mask: Iterable):
		"""
		Returns a new SparseDataFrame that only contains the columns that are True in the mask.
		"""
		if not any(mask):
			warnings.warn('No columns selected.')
		return SparseDataFrame(self.matrix[:, mask], self.index_ids, self.labels, self.column_names[mask], self.dtype)

	def _keep_indices_and_columns(self, idx_mask: Iterable, col_mask: Iterable):
		"""
		Returns a new SparseDataFrame that only contains the rows and columns that are True in the masks.
		"""
		if not any(idx_mask):
			warnings.warn('No indices selected.')
		if not any(col_mask):
			warnings.warn('No columns selected.')
		return SparseDataFrame(self.matrix[idx_mask, col_mask], self.index_ids[idx_mask], self.labels[idx_mask], self.column_names[col_mask], self.dtype)
	
	def update(self, data: dict, use_row_col_names=False):
		"""
		Updates the dataframe with the given data.
		:param data: a dict of (row, col) -> value
		"""
		if not all(type(x) == self.dtype for x in data.values()):
			warnings.warn('Not all values in data have the same dtype as the dataframe. Casting data to dataframe dtype.')
			data = {k: self.dtype(v) for k, v in data.items()}
		if use_row_col_names and not(self.index_ids is not None and self.column_names is not None):
			raise TypeError('index_ids and column_names must be defined for use_row_col_names=True')
		if not use_row_col_names and not all(type(x[0]) == int and type(x[1]) == int and len(x) == 2 for x in data.keys()):
			raise TypeError('keys must be tuples of two ints for use_row_col_names=False')
		dok = self.matrix.todok()
		if use_row_col_names:
			index_map = {idx: i for i, idx in enumerate(self.index_ids)}
			column_map = {col: i for i, col in enumerate(self.column_names)}
		for row_col, value in data.items():
			if use_row_col_names:
				row, col = row_col
				row = index_map[row]
				col = column_map[col]
				row_col = (row, col)
			if value == 0 and row_col in dok:
				dok.pop(row_col)
			else:
				dok[row_col] = value
		return SparseDataFrame(dok.tocsr(), self.index_ids, self.labels, self.column_names, self.dtype)

	def get_label(self, label):
		"""
		Returns a new SparseDataFrame that only contains the rows with the given label.
		Raises an AssertionError if no rows with the given label are found.
		"""
		mask = self.labels == label
		if not any(mask):
			raise ValueError(f"No rows with label {label} found")
		return self._keep_indices(mask)
	
	def unique_labels(self) -> set:
		"""
		Returns the unique labels in the dataframe.
		"""
		return set(self.labels)
	
	def remove_label(self, label):
		"""
		Removes all rows with the given label from the dataframe.
		Raises an AssertionError if no rows with the given label are found.
		"""
		mask = self.labels != label
		return self._keep_indices(mask)

	def select_by_index_ids(self, index_ids: Iterable):
		"""
		Returns a new SparseDataFrame that only contains the rows with the given index ids.
		"""
		if self.index_ids is None:
			raise TypeError('index_ids are not defined for this dataframe.')
		if not isinstance(index_ids, Iterable):
			raise TypeError('index_ids must be an iterable object such as a list, tuple or a numpy array')
		pool = set(index_ids)
		missing_ids = pool - set(self.index_ids)
		if missing_ids:
			raise ValueError('No rows with index ids {} found'.format(missing_ids))
		mask = [idx in pool for idx in self.index_ids]
		return self._keep_indices(mask)
	
	def select_by_labels(self, labels: Iterable):
		"""
		Returns a new SparseDataFrame that only contains the rows with the given labels.
		"""
		if self.labels is None:
			raise TypeError('labels are not defined for this dataframe.')
		if not isinstance(labels, Iterable):
			raise TypeError('labels must be an iterable object such as a list, tuple or a numpy array')
		pool = set(labels)
		missing_labels = pool - set(self.labels)
		if missing_labels:
			raise ValueError('No rows with labels {} found'.format(missing_labels))
		mask = [label in pool for label in self.labels]
		return SparseDataFrame(self.matrix[mask, :], self.index_ids[mask], self.labels[mask])

	def select_by_columns(self, column_names: Iterable):
		"""
		Returns a new SparseDataFrame that only contains the columns with the given names.
		"""
		if self.column_names is None:
			raise TypeError('column names are not defined for this dataframe.')
		if not isinstance(column_names, Iterable):
			raise TypeError('column_names must be an iterable object such as a list, tuple or a numpy array')
		pool = set(column_names)
		missing_columns = pool - set(self.column_names)
		if missing_columns:
			raise ValueError('No columns with names {} found'.format(missing_columns))
		mask = [column_name in pool for column_name in self.column_names]
		return self._keep_columns(mask)
	
	def select_by_index_ids_and_columns(self, index_ids: Iterable, column_names: Iterable):
		"""
		Returns a new SparseDataFrame that only contains the rows with the given index ids and the columns with the given names.
		"""
		if self.index_ids is None:
			raise TypeError('index_ids are not defined for this dataframe.')
		if not isinstance(index_ids, Iterable):
			raise TypeError('index_ids must be an iterable object such as a list, tuple or a numpy array')
		if self.column_names is None:
			raise TypeError('column names are not defined for this dataframe.')
		if not isinstance(column_names, Iterable):
			raise TypeError('column_names must be an iterable object such as a list, tuple or a numpy array')
		pool = set(index_ids)
		missing_ids = pool - set(self.index_ids)
		if missing_ids:
			raise ValueError('No rows with index ids {} found'.format(missing_ids))
		idx_mask = [idx in pool for idx in self.index_ids]
		pool = set(column_names)
		missing_columns = pool - set(self.column_names)
		if missing_columns:
			raise ValueError('No columns with names {} found'.format(missing_columns))
		col_mask = [column_name in pool for column_name in self.column_names]
		return self._keep_indices_and_columns(idx_mask, col_mask)

	def drop_singleton_labels(self):
		"""
		Drops all labels (and their samples) that only occur once in the dataframe.
		"""
		counts:dict = self.get_label_counts()
		mask = [counts[label] > 1 for label in self.labels]
		return self._keep_indices(mask)

	def drop_labels_below_threshold(self, threshold: int):
		"""
		Drops all labels (and their samples) that occur less than threshold times in the dataframe.
		"""
		counts:dict = self.get_label_counts()
		mask = [counts[label] >= threshold for label in self.labels]
		self.matrix = self.matrix[mask, :]
		self.labels = self.labels[mask]
		self.index_ids = self.index_ids[mask]

	def get_label_counts(self) -> dict:
		"""
		Returns a dict of counts for each label.
		"""
		result = {label: 0 for label in self.unique_labels()}
		for label in self.labels:
			result[label] += 1
		return result

	def get_label_to_index_ids(self) -> dict:
		"""
		Returns a dict mapping each label to the ids of the dataframe that have that label.
		"""
		result = {label: [] for label in self.unique_labels()}
		for idx_id, label in zip(self.index_ids, self.labels):
			result[label].append(idx_id)
		return result
	
	def get_label_to_indices(self) -> dict:
		"""
		Returns a dict mapping each label to the indices of the dataframe that have that label.
		"""
		result = {label: [] for label in self.unique_labels()}
		for i, label in zip(range(len(self.labels)), self.labels):
			result[label].append(i)
		return result

	def get_stratified_train_test_split(self, test_size:float) -> tuple:
		"""
		Returns a stratified train-test split of the dataframe.
		:param test_size: the fraction of the dataframe to use for testing.
		"""
		assert 0 < test_size < 1
		counts:dict = self.get_label_counts()
		test_counts = {label: ceil(count * test_size) for label, count in counts.items()}
		train_counts = {label: count - test_counts[label] for label, count in counts.items()}
		if not all([count > 0 for count in test_counts.values()]):
			raise ValueError('Not enough data for split. call drop_singleton_labels before splitting.')
		if not all([count > 0 for count in train_counts.values()]):
			raise ValueError('Not enough data for split. call drop_singleton_labels before splitting.')
		label_to_index_ids = self.get_label_to_index_ids()
		test_split_ids = []
		for label, count in test_counts.items():
			test_split_ids += random.sample(label_to_index_ids[label], count)
		train_split_ids = set(self.index_ids) - set(test_split_ids)
		test_split_ids = set(test_split_ids)
		# rearrange the index ids so that they are in the same order as the original dataframe, to not scramble the data
		train_split_ids = [idx for idx in self.index_ids if idx in train_split_ids]
		test_split_ids = [idx for idx in self.index_ids if idx in test_split_ids]
		return self.select_by_index_ids(train_split_ids), self.select_by_index_ids(test_split_ids)
	
	def _validate_compatiblity(self, other: SparseDataFrame, vertical=True):
		if not isinstance(other, SparseDataFrame):
			raise TypeError('other must be a SparseDataFrame')
		if vertical:
			if not self.matrix.shape[1] == other.matrix.shape[1]:
				raise ValueError('other must have the same number of columns as self')
		else:
			if not self.matrix.shape[0] == other.matrix.shape[0]:
				raise ValueError('other must have the same number of rows as self')
		if (self.column_names is not None and other.column_names is None) or (self.column_names is None and other.column_names is not None):
			raise ValueError('either both or neither of self and other must have column names')
		if (self.labels is not None and other.labels is None) or (self.labels is None and other.labels is not None):
			raise ValueError('either both or neither of self and other must have labels')
		if (self.index_ids is not None and other.index_ids is None) or (self.index_ids is None and other.index_ids is not None):
			raise ValueError('either both or neither of self and other must have index ids')
		if not self.dtype == other.dtype:
			raise ValueError('other must have the same dtype as self')
		if vertical:
			if self.index_ids is not None and other.index_ids is not None and not set(self.index_ids).isdisjoint(set(other.index_ids)):
				raise ValueError('other must have different index ids than self')
			if self.column_names is not None and other.column_names is not None and not all(self.column_names == other.column_names):
				raise ValueError('other must have the same column names as self')
		else:
			if not set(self.column_names).isdisjoint(set(other.column_names)):
				raise ValueError('other must have different column names than self')
			if self.index_ids is not None and other.index_ids is not None and (not all(self.index_ids == other.index_ids)):
				raise ValueError('other must have the same index ids as self')
			if self.labels is not None and other.labels is not None and not all(self.labels == other.labels):
				raise ValueError('other must have the same index ids as self')
			
	def concat_vertically(self, other: SparseDataFrame):
		"""
		Concatenates two SparseDataFrames vertically.
		"""
		self._validate_compatiblity(other, vertical=True)
		
		matrix = vstack((self.matrix, other.matrix)).tocsr()
		index_ids = np.hstack((self.index_ids, other.index_ids))
		labels = np.hstack((self.labels, other.labels))
		return SparseDataFrame(matrix, index_ids, labels, self.column_names)
	
	def concat_horizontally(self, other: SparseDataFrame):
		"""
		Concatenates two SparseDataFrames horizontally.
		"""
		self._validate_compatiblity(other, vertical=False)
		
		matrix = hstack((self.matrix, other.matrix)).tocsr()
		column_names = np.hstack((self.column_names, other.column_names))
		return SparseDataFrame(matrix, self.index_ids, self.labels, column_names)

	def negative_sampling(self, label, n_samples):
		"""
		samples n rows with a label that is different from the given label.
		"""
		if self.labels is None:
			raise ValueError('Labels not defined for this SparseDataFrame.')
		if label not in self.unique_labels():
			raise ValueError('Positive label must exist in labels.')
		pool = [idx for idx,l in zip(self.index_ids, self.labels) if l != label]
		chosen_ids = random.sample(pool, n_samples)
		return self.select_by_index_ids(chosen_ids)
	
	def sample_n_random_labels(self, n: int):
		"""
		Returns a new SparseDataFrame with n random labels from the original dataframe.
		"""
		unique_labels = self.unique_labels()
		if not type(n) == int or n < 0:
			raise TypeError("n must be a positive integer.")
		if n > len(unique_labels):
			raise ValueError(f"n ({n}) too large for SparseDataFrame with {len(unique_labels)} unique labels.")
		label_pool = set(random.sample(unique_labels, n))
		mask = [label in label_pool for label in self.labels]
		return SparseDataFrame(self.matrix[mask, :], self.index_ids[mask], self.labels[mask], self.column_names)
	
	def drop_low_frequency_labels(self, n: int):
		"""
		Drops all labels that occur less than n times in the dataframe.
		"""
		counts = self.get_label_counts()
		mask = [counts[label] >= n for label in self.labels]
		return self._keep_indices(mask)

	def drop_zero_columns(self):
		"""
		Drops all columns that are only zeros.
		"""
		csc = self.matrix.tocsc()
		print(csc[:,0].todense())
		mask = [True if ri > le else False for le, ri in zip(csc.indptr[:-1], csc.indptr[1:])] # don't change "true if ri > le else false" to "ri > le" - for some reason this causes data loss
		print(csc.indptr[0:2])
		return self._keep_columns(mask)

	def random_sample_frac(self, fraction:float):
		"""
		Returns a new SparseDataFrame with a random sample of the rows from the original dataframe.
		"""
		assert 0 < fraction <= 1, 'fraction must be above 0 and up to 1. got {}'.format(fraction)
		selected_indices = np.random.choice(self.index_ids, size=int(len(self.index_ids) * fraction), replace=False)
		return self.select_by_index_ids(selected_indices)
	
	def to_tensorflow(self):
		"""
		Returns a tuple of the matrix, index ids and labels as tensors.
		"""
		coordinate_matrix = self.matrix.tocoo()
		coordinates = np.mat([coordinate_matrix.row, coordinate_matrix.col]).T
		vals = coordinate_matrix.data
		shape = coordinate_matrix.shape
		return tf.SparseTensor(coordinates, vals, shape)

	def to_torch(self):
		"""
		Returns a tuple of the matrix, index ids and labels as tensors.
		"""
		coordinate_matrix = self.matrix.tocoo()
		coordinates = np.array([coordinate_matrix.row, coordinate_matrix.col])
		vals = coordinate_matrix.data
		shape = coordinate_matrix.shape
		return torch.sparse_coo_tensor(coordinates, vals, shape)
