import { sign } from './activation';
import { zip } from './util';

class Perceptron {
  constructor({
    activate = sign,
    addBias = false,
    bias = 1,
    inputs = [],
    learningRate = 0.05,
    weights = []
  }) {
    this.activate = activate;
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
    this.inputs = [];
    inputs.forEach(input => this.inputs.push(input));

    if (this.addBias) {
      this.inputs.push(this.bias);
    }
  }

  assignWeights(weights) {
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
    return this.activate(sum);
  }

  train(targetOutput) {
    if (targetOutput === undefined) {
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
