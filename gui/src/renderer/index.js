import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import './fonts/index.css';

import { HashRouter } from "react-router-dom";
import { MethodProvider } from './logic/context';

import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <HashRouter>
    <MethodProvider>
    <App />
    </MethodProvider>
  </HashRouter>
);