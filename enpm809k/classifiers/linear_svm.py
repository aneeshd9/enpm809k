from typing import Tuple

import numpy as np

def svm_loss_naive(
  W: np.ndarray, # (3073 x 10)
  X: np.ndarray, # (N x 3073)
  y: np.ndarray, # (N x 1)
  reg: float
) -> Tuple[float, np.ndarray]:
  delta = 1.0
  loss = 0.0
  dW = np.zeros(W.shape) # (3073 x 10)
  num_train = X.shape[0] # scalar = N
  num_classes = W.shape[1] # scalar = 10
  
  for i in range(num_train):
    scores_i = X[i].dot(W) # (1 x 10)
    correct_class_score = scores_i[y[i]] # scalar
    for j in range(num_classes):
      if j == y[i]:
        continue
      
      margin = (scores_i[j] - correct_class_score + delta)
      
      if margin > 0.0:
        loss += margin
        dW[:y[i]] -= X[i]
        dW[:j] += X[i]
  
  return loss, dW 
        