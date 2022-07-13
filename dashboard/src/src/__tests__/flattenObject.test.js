import { flattenObject } from '../utils/flattenObject';

test('test flatten object', () => {
  const input = {
    a: 1,
    b: {
      ba: 'world',
      bb: 1.5,
      bc: {
        bd: {
          'a key': 333,
          'another key': {
            x: -1,
            y: true,
          },
        },
      },
      bd: {
        bff: 'abcdefgh',
        orion: 'benchmarks',
      },
      be: 100,
      bf: 'bf',
    },
    c: [10, '2a'],
    d: 'hello',
  };
  const output = flattenObject(input);
  console.log(output);
  const keys = Object.keys(output);
  expect(keys.length).toBe(12);
  expect(output.hasOwnProperty('a')).toBeTruthy();
  expect(output.hasOwnProperty('b.ba')).toBeTruthy();
  expect(output.hasOwnProperty('b.bb')).toBeTruthy();
  expect(output.hasOwnProperty('b.bc.bd.a key')).toBeTruthy();
  expect(output.hasOwnProperty('b.bc.bd.another key.x')).toBeTruthy();
  expect(output.hasOwnProperty('b.bc.bd.another key.y')).toBeTruthy();
  expect(output.hasOwnProperty('b.bd.bff')).toBeTruthy();
  expect(output.hasOwnProperty('b.bd.orion')).toBeTruthy();
  expect(output.hasOwnProperty('b.be')).toBeTruthy();
  expect(output.hasOwnProperty('b.bf')).toBeTruthy();
  expect(output.hasOwnProperty('c')).toBeTruthy();
  expect(output.hasOwnProperty('d')).toBeTruthy();
});
