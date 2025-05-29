import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Layout } from '../../components/layout/Layout';
import { PatientDashboard } from './PatientDashboard';
import { PharmacistDashboard } from './PharmacistDashboard';

export const Dashboard: React.FC = () => {
  const { user } = useAuth();

  return (
    <Layout>
      {user?.role === 'pharmacist' || user?.role === 'admin' ? (
        <PharmacistDashboard />
      ) : (
        <PatientDashboard />
      )}
    </Layout>
  );
};