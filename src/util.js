// combines element pairs in the lists into single tuples
// only returns items with a pair in the other lists
const zip = (...arrays) => {
  if (arrays.some(array => array === null || array === undefined)) return [];

  // find longest of the arrays to avoid extra elements
  // and keep returned elements limited to same-sized tuples
  const shortest = arrays.reduce((a, b) => (a.length < b.length ? a : b));

  // map each element of the shortest array to
  // all elements of each array at index i
  return shortest.map((_, i) => arrays.map(array => array[i]));
};

export { zip };
