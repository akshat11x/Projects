// src/pages/auth/SignupPage.tsx
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { motion } from 'framer-motion';
import { User, Mail, Lock, AlertCircle } from 'lucide-react';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Card, CardHeader, CardContent, CardFooter } from '../../components/ui/Card';
// export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

type SignupFormData = {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
};

export const SignupPage: React.FC = () => {
  const { register, handleSubmit, watch, formState: { errors } } = useForm<SignupFormData>();
  const [signupError, setSignupError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const password = watch('password');

  const onSubmit = async (data: SignupFormData) => {
    setSignupError(null);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: data.name,
          email: data.email,
          password: data.password,
        }),
      });

      const text = await response.text();

      let resData;
      try {
        resData = JSON.parse(text);
      } catch (err) {
        console.error('❌ Non-JSON response from server:', text);
        throw new Error('Server returned an unexpected response');
      }

      if (!response.ok) {
        throw new Error(resData.message || 'Signup failed');
      }

      // Show success message and redirect to login
      const successMessage = 'Account created successfully! Please log in.';
      navigate('/login', { state: { message: successMessage } });
    } catch (error: any) {
      console.error('Signup error:', error);
      setSignupError(error.message || 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="max-w-md w-full">
        <div className="text-center mb-6">
          <h2 className="text-3xl font-extrabold text-gray-900">Create your account</h2>
          <p className="mt-2 text-sm text-gray-600">Join MED-KIT to better manage your health</p>
        </div>

        <Card>
          <CardHeader>
            {signupError && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md flex items-start mb-4">
                <AlertCircle className="h-5 w-5 mr-2 mt-0.5 flex-shrink-0" />
                <span>{signupError}</span>
              </motion.div>
            )}
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <Input
                label="Full Name"
                type="text"
                icon={<User className="h-5 w-5" />}
                error={errors.name?.message}
                fullWidth
                autoComplete="name"
                {...register('name', {
                  required: 'Name is required',
                  minLength: { value: 2, message: 'At least 2 characters' }
                })}
              />

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
                autoComplete="new-password"
                {...register('password', {
                  required: 'Password is required',
                  minLength: { value: 8, message: 'Minimum 8 characters' },
                  pattern: {
                    value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/,
                    message: 'Must include uppercase, lowercase, number, special char'
                  }
                })}
              />

              <Input
                label="Confirm Password"
                type="password"
                icon={<Lock className="h-5 w-5" />}
                error={errors.confirmPassword?.message}
                fullWidth
                autoComplete="new-password"
                {...register('confirmPassword', {
                  required: 'Confirm your password',
                  validate: value => value === password || 'Passwords do not match'
                })}
              />

              <div className="flex items-center">
                <input id="terms" name="terms" type="checkbox" required className="h-4 w-4 text-blue-600 border-gray-300 rounded" />
                <label htmlFor="terms" className="ml-2 text-sm text-gray-700">
                  I agree to the <Link to="/terms" className="text-blue-600 hover:text-blue-500">Terms</Link> and <Link to="/privacy" className="text-blue-600 hover:text-blue-500">Privacy Policy</Link>
                </label>
              </div>

              <Button type="submit" variant="primary" fullWidth size="lg" isLoading={isLoading}>
                Create Account
              </Button>
            </form>
          </CardContent>

          <CardFooter className="flex justify-center">
            <p className="text-sm text-gray-600">
              Already have an account? <Link to="/login" className="font-medium text-blue-600 hover:text-blue-500">Sign in</Link>
            </p>
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  );
};
