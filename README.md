[![Build Status](https://travis-ci.com/tireymorris/tinyneuron.svg?branch=master)](https://travis-ci.com/tireymorris/tinyneuron)

# tinyneuron

An implementation of various neural networks concepts in JavaScript

## Perceptrons

#### Creation/config

Create a perceptron like so:

```
const config = { inputs: [0.25, 0.5, -0.75, 1.0, -1.25] };
const perceptron = new Perceptron(config);
```

A Perceptron has `inputs`, `weights`, a `learningRate`, and an activation function `activation`, which can all be passed into the constructor config. The activation function defaults to the sign function, `const sign = (x) => x >= 0 ? 1 : -1`. The number of weights defaults to inputs length. You can also optionally disable the bias neuron with the config flag `addBias: false`, or tweak the bias output with `bias: value`. The learning rate defaults to 5%, which still allows for dramatic improvements between cycles.

To generate perceptron outputs:

```
const perceptron = new Perceptron({ numWeights: 2, inputs: [1.0, -1.0] });
console.log(perceptron.getOutput()) // -> activation( (Math.random() * 2 - 1) * 1.0 + (Math.random() * 2 - 1) * -1.0)
```

weights are assigned randomly as values between -1 and 1.

To reassign weights and inputs, call `percepton.assignWeights(array)` or `perceptron.assignInputs(array)`

#### Training

To train the perceptron against a target value, run:

```
perceptron.train(targetOutput);
```

This function calculates the predicted output based on current weights and inputs and an error between the target and predicted outputs tempered by the learning rate. It then adjusts its weights to compensate for error. For a large learning rate generally only a few rounds of training at most are necessary.

#### Activation

To use another activation function (e.g. sigmoid):

```
import { activation, Perceptron } from 'tinyneuron';
const perceptron = new Perceptron({ activation: activation.sigmoid });
```

##### Example: AND gate

```
const andNode = new Perceptron({
  addBias: true,
  weights: [1, 1, -1.5]
});

andNode.assignInputs([0, 0]);
andNode.getOutput() === -1;

andNode.assignInputs([0, 1]);
andNode.getOutput() === -1;

andNode.assignInputs([1, 0]);
andNode.getOutput() === -1;

andNode.assignInputs([1, 1]);
andNode.getOutput() === 1;
```
