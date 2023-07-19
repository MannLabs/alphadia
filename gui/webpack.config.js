const path = require('path');
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");

const isProduction = true;

module.exports = {
    entry: './src/renderer/index.js',
    output: {
        filename: 'index.js',
        path: path.resolve(__dirname, 'dist'),
        publicPath: './'
    },
    target: "electron-renderer",
    mode: isProduction ? "production" : "development",
    module: {
       rules: [
         {
           test: /\.jsx?$/,
           exclude: /node_modules/,
           use: {
             loader: "babel-loader",
             options: {
               cacheDirectory: true,
               cacheCompression: false,
               envName: isProduction ? "production" : "development"
             }
           }
         },
        {
            test: /\.css$/,
            use: [
                "style-loader",
                "css-loader"
            ],
            
        },
        {
            test: /\.svg$/,
            use: ["@svgr/webpack"]
        },
        {
          test: /\.(woff|woff2|eot|ttf)$/,
          type: 'asset/resource',
          generator: {
            filename: './fonts/[name][ext]',
        },
        },
        ]
     },
     resolve: {
       extensions: [".js", ".jsx"]
     },
    plugins: [
        new HtmlWebpackPlugin({
            template: path.resolve(__dirname, "./src/renderer/index.html"),
            inject: true
            }),
        new CopyWebpackPlugin({
            patterns: [
                { from: 'src/main/' }
            ]
        })
    ]
};