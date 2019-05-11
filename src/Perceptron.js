import { sign } from './activation';
import { zip } from './util';

class Perceptron {
  constructor({
    activation = sign,
    addBias = false,
    bias = 1,
    inputs = [],
    learningRate = 0.05,
    weights = []
  }) {
    this.activation = activation;
    this.addBias = addBias;
    this.bias = 1;
    this.learningRate = learningRate;

    this.assignInputs(inputs);

    if (weights.length > 0) {
      this.assignWeights(weights);
    } else {
      this.generateWeights();
    }
  }

  assignInputs(inputs) {
    if (
      !inputs instanceof Array ||
      (inputs.length > 0 && inputs.some(input => typeof input !== 'number'))
    ) {
      throw new Error('must pass array of numbers into assignWeights');
    }

    this.inputs = [];
    inputs.forEach(input => this.inputs.push(input));

    if (this.addBias) {
      this.inputs.push(this.bias);
    }
  }

  assignWeights(weights) {
    if (
      !weights instanceof Array ||
      (weights.length > 0 && weights.some(weight => typeof weight !== 'number'))
    ) {
      throw new Error('must pass array of numbers into assignWeights');
    }
    this.weights = [];

    weights.forEach(weight => this.weights.push(weight));
  }

  generateWeights() {
    // Generate weights between -1 and 1
    this.weights = [];

    for (let i = 0; i < this.inputs.length; i++) {
      this.weights.push(Math.random() * 2 - 1);
    }
  }

  getOutput() {
    // calculate weighted sum of inputs and weights
    const sum = zip(this.inputs, this.weights).reduce(
      (acc, [input, weight]) => acc + input * weight,
      0.0
    );

    // run activation function on weighted sum
    return this.activation(sum);
  }

  train(targetOutput) {
    if (targetOutput === undefined || typeof targetOutput !== 'number') {
      throw new Error('must provide targetOutput to Perceptron::train()');
    }
    // retrieve current output
    const guessedOutput = this.getOutput();

    // calculate error between known expected value and guessed output
    const error = this.learningRate * (targetOutput - guessedOutput);

    // adjust weights to compensate for error
    this.weights = this.weights.map(
      (weight, index) => weight + error * this.inputs[index]
    );
  }
}

export default Perceptron;
