import React, { useState } from 'react';
import { FileUp, Loader } from 'lucide-react';
import Button from '../common/Button';

interface ScanResult {
  summary: string;
  full_text: string;
}

const ReportScanner: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
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
      setError('Please select a file to scan');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5001/api/scan/report', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to scan report');
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
      <h2 className="text-2xl font-semibold text-gray-900 mb-6">Health Report Scanner</h2>
      
      <div className="space-y-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            id="report-file"
          />
          <label
            htmlFor="report-file"
            className="cursor-pointer flex flex-col items-center justify-center"
          >
            <FileUp className="h-12 w-12 text-gray-400 mb-3" />
            <span className="text-sm text-gray-600">
              {file ? file.name : 'Click to upload or drag and drop'}
            </span>
            <span className="text-xs text-gray-500 mt-1">
              Supported formats: PNG, JPG, PDF
            </span>
          </label>
        </div>

        <Button
          onClick={handleScan}
          disabled={!file || isLoading}
          isLoading={isLoading}
          fullWidth
        >
          {isLoading ? 'Scanning...' : 'Scan Report'}
        </Button>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {result && (
          <div className="bg-blue-50 border border-blue-200 rounded-md p-4 space-y-4">
            <div>
              <h3 className="text-sm font-medium text-blue-900">Summary</h3>
              <p className="mt-1 text-sm text-blue-700">{result.summary}</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-blue-900">Full Text</h3>
              <p className="mt-1 text-sm text-blue-700 whitespace-pre-wrap">
                {result.full_text}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportScanner;