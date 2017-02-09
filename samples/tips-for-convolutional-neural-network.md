Tips
====

Techniques that could be used during CNN modeling:
  - Transform training data to range [-1, 1]
  - Relu - activation function that enforces non linear solutions by zeroing negative values from signals
  - Pooling - resample down image after convolution - MaxPool, AvgPool
  - Dropout - randomly zero some signals on network in order to enforce multiple solution paths during training - make the network solution more resilient to noise and overfitting
  - L2 loss beta compensation - minimize ...
  - Learn Rate Decay - exponential learn decay - AdamOptimizer for automatic learn rate optimization
  - Stop training as soon as solution stops improving to avoid overfitting
  
