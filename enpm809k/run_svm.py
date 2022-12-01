from datasets.data_utils import load_cifar10
from typing import Tuple

import os
import numpy as np

def data_setup(
  num_training: int=49000,
  num_test: int=1000,
  num_validation: int=1000,
  flatten: bool=True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
  
  num_dev = 500

  X_val = X_train[range(num_training, num_training + num_validation)]
  y_val = y_train[range(num_training, num_training + num_validation)]
  X_train = X_train[range(num_training)]
  y_train = y_train[range(num_training)]
  mask = np.random.choice(num_training, num_dev, replace=False)
  X_dev = X_train[mask]
  y_dev = y_train[mask]
  X_test = X_test[range(num_test)]
  y_test = y_test[range(num_test)]
  
  if flatten:
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

  return X_train, y_train, X_test, y_test, X_val, y_val, X_dev, y_dev
  
def svm():
  X_train, y_train, X_test, y_test, X_val, y_val, X_dev, y_dev = data_setup()
  
  # Subtract mean and add '1' row so bias becomes part of the weight matrix
  mean = np.mean(X_train, axis=0)
  X_train -= mean
  X_test -= mean
  X_val -= mean
  X_dev -= mean
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
  X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
  
if __name__ == '__main__':
  svm()