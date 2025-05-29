import React, { useState } from 'react';
import Input from '../common/Input';
import Button from '../common/Button';

interface TwoFactorAuthProps {
  onVerify: (otp: string) => Promise<void>;
  onResend: () => Promise<void>;
}

const TwoFactorAuth: React.FC<TwoFactorAuthProps> = ({ onVerify, onResend }) => {
  const [otp, setOtp] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isResending, setIsResending] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      await onVerify(otp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Verification failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResend = async () => {
    setIsResending(true);
    try {
      await onResend();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resend code');
    } finally {
      setIsResending(false);
    }
  };

  return (
    <div className="w-full max-w-md">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <Input
            id="otp"
            name="otp"
            type="text"
            label="Verification Code"
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            placeholder="Enter 6-digit code"
            required
            fullWidth
            maxLength={6}
            pattern="[0-9]{6}"
          />
        </div>

        {error && (
          <div className="rounded-md bg-red-50 p-4">
            <div className="flex">
              <div className="text-sm text-red-700">{error}</div>
            </div>
          </div>
        )}

        <div className="flex flex-col space-y-3">
          <Button
            type="submit"
            fullWidth
            isLoading={isLoading}
          >
            Verify Code
          </Button>

          <Button
            type="button"
            variant="text"
            fullWidth
            onClick={handleResend}
            isLoading={isResending}
            disabled={isResending}
          >
            Resend Code
          </Button>
        </div>
      </form>
    </div>
  );
};

export default TwoFactorAuth;