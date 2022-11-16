from classifiers.knn import KNearestNeighbor
from datasets.data_utils import load_cifar10
from typing import Tuple

import os
import numpy as np

def data_setup(
  num_training: int=None, num_test: int=None, flatten: bool=True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  cifar10_root = os.path.join(
    os.path.dirname(__file__),
    'datasets', 'cifar-10-batches-py'
  )
  
  if not os.path.exists(cifar10_root):
    print('Dataset is not downloaded. Attempting download...')
    try:
      script_dir = os.path.join(
        os.path.dirname(__file__),
        'datasets'
      )
      success = os.system(f'cd {script_dir} && ./get_dataset.sh')
      if success == 0:
        print('Download finished!')
      else:
        raise RuntimeError('Failed to run donwload script. See terminal output.')
    except Exception as e:
      print('Something went wrong while downloading.')
      print(e)
      exit(1)
  
  X_train, y_train, X_test, y_test = load_cifar10(cifar10_root)
  
  if num_training:
    X_train = X_train[range(num_training)]
    y_train = y_train[range(num_training)]
  if num_test:
    X_test = X_test[range(num_test)]
    y_test = y_test[range(num_test)]
  if flatten:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
  
  return X_train, y_train, X_test, y_test

def knn():
  def knn_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    num_correct = np.sum(pred == gt)
    accuracy = float(num_correct) / gt.shape[0]
    print(f'Got {num_correct} / {gt.shape[0]} correct => accuracy {accuracy}')
    return accuracy
  
  X_train, y_train, X_test, y_test = data_setup(num_training=5000, num_test=500)
  print(f'Training data shape: {X_train.shape}')
  print(f'Training labels shape: {y_train.shape}')
  print(f'Test data shape: {X_test.shape}')
  print(f'Test labels shape: {y_test.shape}')
  
  classifier = KNearestNeighbor()
  classifier.train(X_train, y_train)
  
  dists = classifier.compute_distances_two_loops(X_test)
  print(dists.shape)
  
  y_test_pred = classifier.predict_labels(dists, 1)
  knn_accuracy(y_test_pred, y_test)
  
  y_test_pred = classifier.predict_labels(dists, 5)
  knn_accuracy(y_test_pred, y_test)

if __name__ == '__main__':
  knn()