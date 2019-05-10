import { sign } from './activation';
import { zip } from './util';

class Perceptron {
  constructor({
    activate = sign,
    addBias = true,
    inputs = [],
    learningRate = 0.05
  }) {
    this.activate = activate;
    this.addBias = addBias;
    this.inputs = inputs;
    this.learningRate = learningRate;

    this.generateWeights();
  }

  generateWeights() {
    // Generate weights between -1 and 1
    this.weights = [];

    for (let i = 0; i < this.inputs.length; i++) {
      this.weights.push(Math.random() * 2 - 1);
    }

    if (this.addBias) {
      this.weights.push(1);
      this.inputs.push(1);
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
