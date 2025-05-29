import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '../context/AuthContext';
import PrivateRoute from '../utils/PrivateRoute';
import AuthenticationPage from '../pages/AuthenticationPage';
import Dashboard from '../pages/Dashboard';
import ReportScanner from '../components/scanning/ReportScanner';
import MedicineScanner from '../components/scanning/MedicineScanner';

const AppRoutes: React.FC = () => {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<Navigate to="/login\" replace />} />
          <Route path="/login" element={<AuthenticationPage />} />
          <Route path="/signup" element={<AuthenticationPage />} />
          
          {/* Protected routes */}
          <Route element={<PrivateRoute />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/scan/report" element={<ReportScanner />} />
            <Route path="/scan/medicine" element={<MedicineScanner />} />
            <Route path="/appointments" element={<Dashboard />} />
            <Route path="/medications" element={<Dashboard />} />
            <Route path="/records" element={<Dashboard />} />
            <Route path="/metrics" element={<Dashboard />} />
            <Route path="/settings" element={<Dashboard />} />
          </Route>
          
          {/* Fallback route */}
          <Route path="*" element={<Navigate to="/\" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
};

export default AppRoutes;