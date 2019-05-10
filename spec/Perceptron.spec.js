import Perceptron from '../src/Perceptron';

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
    test('with sign activation function returns 1 for positive weights and activated inpus', () => {
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
        learningRate: 0.005,
        numWeights: 3
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
});
