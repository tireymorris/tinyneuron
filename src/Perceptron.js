import { sign } from './activation';
import { zip } from './util';

class Perceptron {
  constructor({ numWeights = 1, inputs = [], activate = sign }) {
    this.weights = [];
    this.inputs = inputs;
    this.activate = activate;

    this.generateWeights(numWeights);
  }

  generateWeights(numWeights) {
    // Generate weights between -1 and 1
    for (let i = 0; i < numWeights; i++) {
      this.weights.push(Math.random() * 2 - 1);
    }
  }

  getOutput() {
    const sum = zip(this.inputs, this.weights).reduce(
      (acc, [input, weight]) => acc + input * weight,
      0.0
    );
    return this.activate(sum);
  }
}

export default Perceptron;
