import { sign, sigmoid } from '../src/activation';

describe('activation', () => {
  describe('sign', () => {
    test('returns 1 for n=0.5', () => {
      expect(sign(0.5)).toEqual(1);
    });
    test('returns -1 for n=0.5', () => {
      expect(sign(-0.5)).toEqual(-1);
    });
  });
  describe('sigmoid', () => {
    test('returns 0.5 for n=0', () => {
      expect(sigmoid(0)).toEqual(0.5);
    });
    test('correctly calculates sigmoid(1)', () => {
      expect(sigmoid(1)).toBeGreaterThan(0.73105857863);
      expect(sigmoid(1)).toBeLessThan(0.73105857864);
    });
  });
});
