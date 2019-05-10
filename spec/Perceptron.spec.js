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

      const perceptron = new Perceptron({ numWeights: 2, inputs: [1.0, 1.0] });
      expect(perceptron.getOutput()).toEqual(1);

      Math.random = random;
    });

    test('with sign activation function returns 0 for mixed inputs and negative weights', () => {
      const random = Math.random;
      Math.random = jest.fn().mockImplementation(() => -0.5);

      const perceptron = new Perceptron({ numWeights: 2, inputs: [1.0, 0.0] });
      expect(perceptron.getOutput()).toEqual(-1);

      Math.random = random;
    });
  });
});
