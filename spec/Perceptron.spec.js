import Perceptron from '../src/Perceptron';
import { sigmoid, sign } from '../src/activation';

describe('Perceptron', () => {
  describe('generateWeights', () => {
    test('should have weights between -1 and 1', () => {
      const perceptron = new Perceptron({});
      perceptron.generateWeights(200);

      perceptron.weights.forEach(weight => {
        expect(weight).toBeLessThanOrEqual(1);
        expect(weight).toBeGreaterThanOrEqual(-1);
      });
    });
  });

  describe('getOutput', () => {
    test('with sign activation function returns 1 for positive weights and activationd inpus', () => {
      const random = Math.random;
      Math.random = jest.fn().mockImplementation(() => 0.5);

      const perceptron = new Perceptron({ inputs: [1.0, 1.0] });
      expect(perceptron.getOutput()).toEqual(1);

      Math.random = random;
    });

    test('with sign activation function returns 0 for mixed inputs and negative weights', () => {
      const random = Math.random;
      Math.random = jest.fn().mockImplementation(() => -0.5);

      const perceptron = new Perceptron({ inputs: [1.0, 0.0] });
      expect(perceptron.getOutput()).toEqual(-1);

      Math.random = random;
    });
  });

  describe('train', () => {
    test('adjusts perceptron weights during training based on target value', () => {
      // mock Math.random to force suboptimal weights to begin with
      const random = Math.random;
      Math.random = jest.fn().mockImplementation(() => 1.0);

      // hypothetical perceptron that returns -1 if point
      // represented by inputs x, y, z has negative z
      const perceptron = new Perceptron({
        addBias: false,
        inputs: [1.25, 1.5, -5.75], // x, y, z
        learningRate: 0.005
      });

      expect(perceptron.getOutput()).toEqual(-1);

      Math.random = random;

      // train for 10 rounds with small learning rate (requires more rounds)
      for (let i = 0; i < 10; i++) {
        perceptron.train(1);
      }

      expect(perceptron.getOutput()).toEqual(1);

      Math.random = random;
    });
  });

  describe('general', () => {
    test('assigns weights correctly', () => {
      const perceptron = new Perceptron({ weights: [1, 0, 1] });
      expect(perceptron.weights.length).toBeGreaterThan(0);
    });
    test('assigns inputs correctly', () => {
      const perceptron = new Perceptron({ inputs: [1, 0, 1] });
      expect(perceptron.inputs.length).toBeGreaterThan(0);
    });
    test('can represent boolean AND', () => {
      // bias and weight values found here:
      // https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
      // https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1
      const andNode = new Perceptron({
        addBias: true,
        weights: [1, 1, -1.5]
      });

      andNode.assignInputs([0, 0]);
      expect(andNode.getOutput()).toEqual(-1);

      andNode.assignInputs([0, 1]);
      expect(andNode.getOutput()).toEqual(-1);

      andNode.assignInputs([1, 0]);
      expect(andNode.getOutput()).toEqual(-1);

      andNode.assignInputs([1, 1]);
      expect(andNode.getOutput()).toEqual(1);
    });
    test('can represent boolean OR', () => {
      // bias and weight values found here:
      // https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
      // https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1
      const andNode = new Perceptron({
        addBias: true,
        weights: [1, 1, -1]
      });

      andNode.assignInputs([0, 0]);
      expect(andNode.getOutput()).toEqual(-1);

      andNode.assignInputs([0, 1]);
      expect(andNode.getOutput()).toEqual(1);

      andNode.assignInputs([1, 0]);
      expect(andNode.getOutput()).toEqual(1);

      andNode.assignInputs([1, 1]);
      expect(andNode.getOutput()).toEqual(1);
    });
  });
});
