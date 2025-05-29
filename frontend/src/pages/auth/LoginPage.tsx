// src/pages/auth/LoginPage.tsx
import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { motion } from 'framer-motion';
import { Mail, Lock, LogIn, AlertCircle } from 'lucide-react';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Card, CardHeader, CardContent, CardFooter } from '../../components/ui/Card';
import { useAuth } from '../../contexts/AuthContext';

type LoginFormData = {
  email: string;
  password: string;
};

export const LoginPage: React.FC = () => {
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormData>();
  const [loginError, setLoginError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isAuthenticated } = useAuth();

  // If user is already authenticated, redirect to dashboard
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  // Get the return URL from location state or default to dashboard
  const from = location.state?.from?.pathname || '/dashboard';

  // Set success message if coming from signup
  useEffect(() => {
    if (location.state?.message) {
      setSuccessMessage(location.state.message);
      // Clear the message from location state
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  const onSubmit = async (data: LoginFormData) => {
    setLoginError(null);
    setIsLoading(true);
    
    try {
      const success = await login(data.email, data.password);
      
      if (success) {
        // Redirect to the page they tried to visit or dashboard
        navigate(from, { replace: true });
      } else {
        setLoginError('Invalid credentials. Please try again.');
      }
    } catch (error) {
      setLoginError('An error occurred during login. Please try again.');
      console.error('Login error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="max-w-md w-full">
        <div className="text-center mb-6">
          <h2 className="text-3xl font-extrabold text-gray-900">Welcome back</h2>
          <p className="mt-2 text-sm text-gray-600">Sign in to your MED-KIT account</p>
        </div>

        <Card>
          <CardHeader>
            {successMessage && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-md flex items-start mb-4">
                <span>{successMessage}</span>
              </motion.div>
            )}
            {loginError && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md flex items-start mb-4">
                <AlertCircle className="h-5 w-5 mr-2 mt-0.5 flex-shrink-0" />
                <span>{loginError}</span>
              </motion.div>
            )}
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <Input
                label="Email Address"
                type="email"
                icon={<Mail className="h-5 w-5" />}
                error={errors.email?.message}
                fullWidth
                autoComplete="email"
                {...register('email', {
                  required: 'Email is required',
                  pattern: { value: /\S+@\S+\.\S+/, message: 'Invalid email format' }
                })}
              />

              <Input
                label="Password"
                type="password"
                icon={<Lock className="h-5 w-5" />}
                error={errors.password?.message}
                fullWidth
                autoComplete="current-password"
                {...register('password', {
                  required: 'Password is required',
                  minLength: { value: 6, message: 'At least 6 characters' }
                })}
              />

              <div className="flex items-center justify-between">
                <label className="flex items-center text-sm text-gray-700">
                  <input type="checkbox" className="mr-2 h-4 w-4 text-blue-600" /> Remember me
                </label>
                <Link to="/forgot-password" className="text-sm text-blue-600 hover:text-blue-500">Forgot your password?</Link>
              </div>

              <Button type="submit" variant="primary" fullWidth size="lg" isLoading={isLoading} icon={<LogIn className="h-5 w-5" />}>
                Sign in
              </Button>
            </form>
          </CardContent>

          <CardFooter className="flex justify-center">
            <p className="text-sm text-gray-600">Don't have an account? <Link to="/signup" className="font-medium text-blue-600 hover:text-blue-500">Sign up</Link></p>
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  );
};
