import React, { useState } from 'react';
import { Camera, CheckCircle, XCircle } from 'lucide-react';
import Button from '../common/Button';

interface MedicineScanResult {
  is_authentic: boolean;
  medicine_data?: {
    name: string;
    manufacturer: string;
    batch_number: string;
    expiry_date: string;
  };
}

const MedicineScanner: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<MedicineScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleScan = async () => {
    if (!file) {
      setError('Please select a QR code to scan');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5001/api/scan/medicine', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to verify medicine');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-semibold text-gray-900 mb-6">Medicine Authenticity Scanner</h2>
      
      <div className="space-y-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            id="medicine-qr"
          />
          <label
            htmlFor="medicine-qr"
            className="cursor-pointer flex flex-col items-center justify-center"
          >
            <Camera className="h-12 w-12 text-gray-400 mb-3" />
            <span className="text-sm text-gray-600">
              {file ? file.name : 'Upload QR code image or take a photo'}
            </span>
            <span className="text-xs text-gray-500 mt-1">
              Position the QR code clearly in the frame
            </span>
          </label>
        </div>

        <Button
          onClick={handleScan}
          disabled={!file || isLoading}
          isLoading={isLoading}
          fullWidth
        >
          {isLoading ? 'Verifying...' : 'Verify Medicine'}
        </Button>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {result && (
          <div className={`${
            result.is_authentic ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
          } border rounded-md p-4`}>
            <div className="flex items-center mb-4">
              {result.is_authentic ? (
                <CheckCircle className="h-6 w-6 text-green-600 mr-2" />
              ) : (
                <XCircle className="h-6 w-6 text-red-600 mr-2" />
              )}
              <span className={`font-medium ${
                result.is_authentic ? 'text-green-900' : 'text-red-900'
              }`}>
                {result.is_authentic ? 'Authentic Medicine' : 'Potentially Counterfeit'}
              </span>
            </div>

            {result.medicine_data && (
              <div className="space-y-2">
                <p className="text-sm">
                  <span className="font-medium">Name:</span> {result.medicine_data.name}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Manufacturer:</span> {result.medicine_data.manufacturer}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Batch Number:</span> {result.medicine_data.batch_number}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Expiry Date:</span> {result.medicine_data.expiry_date}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MedicineScanner;