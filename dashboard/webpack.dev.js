/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2018, 2020                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const TsconfigPathsPlugin = require('tsconfig-paths-webpack-plugin');


//use this string to host the local proxy version of the application
//note - point your browser to http://localhost:8080/proxy/ to find the web app!
const localPath = '/proxy';
//do not modify this file directly to 'proxy' a remote environment - instead export VISION_SERVICE_API (see the readme!)

module.exports = {
  mode: 'development',
  devtool: 'eval-source-map',
  // if access to original source code isn't needed during development,
  // compilation can be sped up by temporarily changing to 'eval' or 'cheap-eval-source-map'
  entry: {
    app: './src/index.js'
  },
  plugins: [
    new CleanWebpackPlugin(),
    new HtmlWebpackPlugin({
      title: 'Orion Console',
      template: path.resolve(__dirname, './src/index.html'),
      favicon: path.resolve(__dirname, './src/images/bee.png'),
      inject: true,
      hash: true
    }),
    new webpack.IgnorePlugin(/canvas|jsdom|xmldom/),
  ],
  output: {
    filename: 'vision-ui.bundle.[hash:8].js',
    path: path.resolve(__dirname, './dist'),
    //publicPath: "./public/"
  },
  resolve: {
    alias: {
      api: path.resolve(__dirname, './src/api.js'),
      images: path.resolve(__dirname, './src/images'),
      utils: path.resolve(__dirname, './src/utilities')
    },
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
    modules: [path.join(__dirname, "./src/components"), "node_modules"],
    plugins: [new TsconfigPathsPlugin({ configFile: path.join(__dirname, "./tsconfig.json") })],    
  },
  devServer: {
    //contentBase: path.join(__dirname, "./dist"),
    contentBase: path.join(__dirname, "./"),
    //we use publicPath here to discover absolute vs relative path bugs in URLs.
    publicPath: localPath,
    compress: true,
    disableHostCheck: true,
    stats: "errors-only",
    historyApiFallback: true,
  },
  module: {
    rules: [     
      {
        test: /\.css$/,
        use: [ 'style-loader', 'css-loader' ]
      },
      {
        test: /\.scss$/,
        use: [
          { loader: 'style-loader' }, // style-loader is fine for dev builds
          { loader: 'css-loader' },
          {
            loader: 'postcss-loader',
            options: {
              plugins: [ require('autoprefixer') ],
            },
          },
          { loader: 'sass-loader' }
        ]
      },
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        loader: 'babel-loader'
      },
      {
        test: /\.(ts|tsx)$/,
        exclude: /node_modules/,
        loader: require.resolve('babel-loader'),
        options: {
          customize: require.resolve(
            'babel-preset-react-app/webpack-overrides'
          ),
          
          plugins: [
            [
              require.resolve('babel-plugin-named-asset-import'),
              {
                loaderMap: {
                  svg: {
                    ReactComponent: '@svgr/webpack?-svgo,+ref![path]',
                  },
                },
              },
            ],
          ],
        },
      }, 
      {
        test: /\.(png|jpe?g|gif|eot|woff|woff2|ttf|json|ico)$/,
        loader: 'url-loader'
      },
      {
        test: /\.svg$/,
        loader: 'svg-sprite-loader',
        options: {
          extract: false
        }
      }
    ]
  },
  target: 'web',
  externals: {
    //'canvas-prebuilt': 'undefined',
    //'canvas': 'undefined',
    'jsdom/lib/jsdom/utils': JSON.stringify({ Canvas: null }),
    'jsdom/lib/jsdom/living/generated/utils': JSON.stringify({ implForWrapper: null }),
    'jsdom': 'null',
    'xmldom': JSON.stringify({ DOMParser: null }),
  },
};
