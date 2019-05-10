import { sign } from '../src/activation';

describe('activation', () => {
  describe('sign', () => {
    test('returns 1 for n=0.5', () => {
      expect(sign(0.5)).toEqual(1);
    });
    test('returns -1 for n=0.5', () => {
      expect(sign(-0.5)).toEqual(-1);
    });
  });
});
