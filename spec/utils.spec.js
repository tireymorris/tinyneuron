import { zip } from '../src/util';

describe('util', () => {
  describe('zip', () => {
    test('zips two lists together into one list of tuple pairs', () => {
      expect(zip(['a', 'c', 'e'], ['b', 'd', 'f'])).toEqual([
        ['a', 'b'],
        ['c', 'd'],
        ['e', 'f']
      ]);
    });
    test('ignores extra elements', () => {
      expect(zip([1, 2], [3, 4, 5])).toEqual([[1, 3], [2, 4]]);
    });
  });
});
