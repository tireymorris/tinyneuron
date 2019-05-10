[![Build Status](https://travis-ci.com/tireymorris/tinyneuron.svg?branch=master)](https://travis-ci.com/tireymorris/tinyneuron)

# tinyneuron 
An implementation of various neural networks concepts in JavaScript

## Usage
Create a perceptron like so:
```
const perceptron = new Perceptron({ numWeights: 5, inputs: [0.25, 0.5, -0.75, 1.0, -1.25] });
```
A Perceptron has `weights`, `inputs`, and an activation function, `activate`, which can all be passed into the constructor. The activation function defaults to the sign function, `const sign = (x) => x >= 0 ? 1 : -1`. The number of weights defaults to 1. 

To generate perceptron outputs:

```
const perceptron = new Perceptron({ numWeights: 2, inputs: [1.0, -1.0] });
console.log(perceptron.getOutput()) // -> activate( (Math.random() * 2 - 1) * 1.0 + (Math.random() * 2 - 1) * -1.0)
```

weights are assigned randomly as values between -1 and 1, but this may change/become configurable in future revisions.
