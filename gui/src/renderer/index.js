import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import './fonts/index.css';

import { HashRouter } from "react-router-dom";
import { GitHubProvider, MethodProvider } from './logic/context';
import { ProfileProvider } from './logic/profile';

import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <HashRouter>
    <GitHubProvider>
      <MethodProvider>
        <ProfileProvider>
          <App />
        </ProfileProvider>
      </MethodProvider>
    </GitHubProvider>
  </HashRouter>
);
