import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from '../contexts/AuthContext';

import { LandingPage } from '../pages/LandingPage';
import { LoginPage } from '../pages/auth/LoginPage';
import { SignupPage } from '../pages/auth/SignupPage';
// import { Dashboard } from '../pages/dashboard/Dashboard';
import { ReportUploadPage } from '../pages/ReportUploadPage';
import { MedicineValidatorPage } from '../pages/MedicineValidatorPage';
import { PatientDashboard } from '../pages/dashboard/PatientDashboard';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent align-[-0.125em]"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    // Redirect to login but save the attempted URL
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};

const AuthRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent align-[-0.125em]"></div>
      </div>
    );
  }

  if (isAuthenticated) {
    // If user is already authenticated, redirect to dashboard
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

export const AppRoutes: React.FC = () => {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<LandingPage />} />
          
          {/* Auth Routes - Redirect to dashboard if already logged in */}
          <Route path="/login" element={
            <AuthRoute>
              <LoginPage />
            </AuthRoute>
          } />
          <Route path="/signup" element={
            <AuthRoute>
              <SignupPage />
            </AuthRoute>
          } />

          {/* Protected Routes - Require Authentication */}
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <PatientDashboard />
            </ProtectedRoute>
          } />
          <Route path="/upload" element={
            <ProtectedRoute>
              <ReportUploadPage />
            </ProtectedRoute>
          } />
          <Route path="/validator" element={
            <ProtectedRoute>
              <MedicineValidatorPage />
            </ProtectedRoute>
          } />

          {/* Public Routes with Optional Authentication */}
          <Route path="/upload-report" element={<ReportUploadPage />} />
          <Route path="/validate-medicine" element={<MedicineValidatorPage />} />

          {/* Fallback - Redirect to landing page */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
};