import React from 'react';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { HelmetProvider, Helmet } from 'react-helmet-async';
import 'aos/dist/aos.css';

import Layout from './layout/Layout';
import Home from './pages/Home';
import InsuranceForm from './pages/InsuranceForm';

const App = () => {
  return (
    <HelmetProvider>
      < Helmet titleTemplate='%s | Insurance Cost Predictor' defaultTitle='Insurance Cost Predictor' />
      < BrowserRouter>
        <Layout>
          <Routes>
            <Route exact path='/' element={<Home />} />
            <Route exact path='/insurance-form' element={<InsuranceForm />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </HelmetProvider>
  )
}

export default App;

