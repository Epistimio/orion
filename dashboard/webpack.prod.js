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
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
//const TerserJSPlugin = require('terser-webpack-plugin');
const OptimizeCSSAssetsPlugin = require('optimize-css-assets-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const TsconfigPathsPlugin = require('tsconfig-paths-webpack-plugin');

module.exports = {
  mode: 'production',
  entry: {
    app: './src/index.js'
  },
  optimization: {
    //minimizer: [new TerserJSPlugin({}), new OptimizeCSSAssetsPlugin({})],
    minimizer: [new OptimizeCSSAssetsPlugin({})],
  },
  plugins: [
    new CleanWebpackPlugin(),
    new HtmlWebpackPlugin({
      title: 'Orion Console',
      template: path.resolve(__dirname, './src/index.html'),
      favicon: path.resolve(__dirname, './src/images/bee.png'),
      inject: true,
      minify: true
    }),
    new MiniCssExtractPlugin({
      filename: 'vision-ui.styles.[hash].js',
      //'vision-ui.styles.[contenthash:8].css',
      chunkFilename: 'vision-ui.styles.[hash:8].chunk.css',
      ignoreOrder: false
    }),
    new webpack.IgnorePlugin(/canvas|jsdom|xmldom/),
  ],
  output: {
    filename: 'vision-ui.bundle.[hash:8].js',
    path: path.resolve(__dirname, './dist')
    //publicPath: 'somepath' - note - this is not set to allow the app to live at a relative URL
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
  module: {
    rules: [     
      {
        test: /\.css$/,
        use: [ MiniCssExtractPlugin.loader, 'css-loader' ]
      },
      {
        test: /\.scss$/,
        use: [
          { loader: MiniCssExtractPlugin.loader },
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
        test: /\.(png|jpe?g|gif|eot|woff|woff2|ttf)$/,
        loader: 'url-loader?limit=8000&name=[name].[ext]?[hash:8]'
      },
      {
        loader: require.resolve('file-loader'),
        // Exclude `js` files to keep "css" loader working as it injects
        // its runtime that would otherwise be processed through "file" loader.
        // Also exclude `html` and `json` extensions so they get processed
        // by webpacks internal loaders.
        exclude: [/\.(js|mjs|jsx|ts|tsx|css|scss)$/, /\.html$/, /\.json$/,],
        options: {
          name: 'static/media/[name].[hash:8].[ext]',
        },
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
