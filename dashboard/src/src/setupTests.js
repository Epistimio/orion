import { configure } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';
// Useful to test plotly (ref: https://stackoverflow.com/a/62768292)
import 'jest-canvas-mock';
// this adds jest-dom's custom assertions
import '@testing-library/jest-dom';
configure({ adapter: new Adapter() });
// Increase test timeout to support long tests
jest.setTimeout(300000);
// Add necessary mock to test plotly (ref: https://github.com/plotly/react-plotly.js/issues/115#issuecomment-448687417)
window.URL.createObjectURL = function() {};
