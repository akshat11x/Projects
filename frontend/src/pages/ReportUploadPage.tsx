import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, Check, X, Download, Languages, Plus } from 'lucide-react';
import { Layout } from '../components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

export const ReportUploadPage: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'complete' | 'error'>('idle');
  const [reportSummary, setReportSummary] = useState<string | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      setUploadedFile(file);
      setUploadStatus('uploading');
      
      // Simulate file upload and processing
      setTimeout(() => {
        setUploadStatus('processing');
        
        // Simulate AI processing
        setTimeout(() => {
          setUploadStatus('complete');
          setReportSummary(`
            Based on your blood test results from ${new Date().toLocaleDateString()}, here's a summary:

            • Cholesterol: 210 mg/dL (Slightly elevated)
            • HDL Cholesterol: 60 mg/dL (Optimal)
            • LDL Cholesterol: 130 mg/dL (Borderline high)
            • Triglycerides: 150 mg/dL (Normal)
            • Blood Glucose (fasting): 95 mg/dL (Normal)
            • Hemoglobin A1C: 5.6% (Normal)
            
            Your results indicate slightly elevated total cholesterol and LDL cholesterol levels. This may increase your risk for heart disease if not addressed. Consider dietary changes to reduce saturated fat intake and increase physical activity. Your other values are within normal ranges.
            
            Recommended follow-up: Schedule a consultation with your primary care physician within the next 3 months to discuss cholesterol management strategies.
          `);
        }, 3000);
      }, 1500);
    }
  };

  const resetUpload = () => {
    setUploadedFile(null);
    setUploadStatus('idle');
    setReportSummary(null);
  };

  // Animation variants
  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  return (
    <Layout>
      <motion.div 
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        className="max-w-4xl mx-auto"
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Health Report Upload</h1>
        <p className="text-lg text-gray-600 mb-8">
          Upload your medical reports and get AI-powered summaries and insights in simple language.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Upload Health Report</CardTitle>
                <CardDescription>
                  We accept PDF, JPG, and PNG files of lab reports, prescriptions, or any medical documents
                </CardDescription>
              </CardHeader>
              <CardContent>
                {uploadStatus === 'idle' && (
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
                    <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-700 mb-4">Drag and drop your file here, or click to browse</p>
                    <p className="text-sm text-gray-500 mb-6">Supported formats: PDF, JPG, PNG (Max 10MB)</p>
                    <input
                      type="file"
                      id="file-upload"
                      className="hidden"
                      accept=".pdf,.jpg,.jpeg,.png"
                      onChange={handleFileUpload}
                    />
                    <label htmlFor="file-upload">
                      <Button variant="primary" icon={<FileText className="h-5 w-5 mr-2" />} as="span">
                        Select File
                      </Button>
                    </label>
                  </div>
                )}
                
                {(uploadStatus === 'uploading' || uploadStatus === 'processing') && (
                  <div className="border rounded-lg p-6">
                    <div className="flex items-center mb-4">
                      <div className="mr-4">
                        <FileText className="h-10 w-10 text-blue-500" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-medium">{uploadedFile?.name}</h3>
                        <p className="text-sm text-gray-500">{Math.round((uploadedFile?.size || 0) / 1024)} KB</p>
                      </div>
                      <button 
                        onClick={resetUpload} 
                        className="p-2 text-gray-400 hover:text-gray-600"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    
                    <div className="mb-4">
                      <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500 rounded-full"
                          style={{ 
                            width: uploadStatus === 'uploading' ? '50%' : '90%',
                            transition: 'width 1s ease-in-out'
                          }}
                        ></div>
                      </div>
                    </div>
                    
                    <p className="text-center text-gray-600">
                      {uploadStatus === 'uploading' ? 'Uploading file...' : 'Processing with AI...'}
                    </p>
                  </div>
                )}
                
                {uploadStatus === 'complete' && reportSummary && (
                  <div className="border rounded-lg p-6">
                    <div className="flex items-center mb-4">
                      <div className="mr-4 bg-green-100 p-2 rounded-full">
                        <Check className="h-6 w-6 text-green-600" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-medium">Analysis Complete</h3>
                        <p className="text-sm text-gray-500">{uploadedFile?.name}</p>
                      </div>
                      <button 
                        onClick={resetUpload} 
                        className="p-2 text-gray-400 hover:text-gray-600"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    
                    <div className="bg-gray-50 p-4 rounded-lg mb-4">
                      <h3 className="font-medium text-gray-900 mb-2">Report Summary</h3>
                      <div className="text-gray-700 whitespace-pre-line">
                        {reportSummary}
                      </div>
                    </div>
                    
                    <div className="flex flex-wrap gap-2">
                      <Button variant="outline" icon={<Download className="h-4 w-4 mr-2" />}>
                        Download Summary
                      </Button>
                      <Button variant="outline" icon={<Languages className="h-4 w-4 mr-2" />}>
                        Translate
                      </Button>
                      <Button variant="primary" icon={<Plus className="h-4 w-4 mr-2" />}>
                        Add to Medical History
                      </Button>
                    </div>
                  </div>
                )}
                
                {uploadStatus === 'error' && (
                  <div className="border border-red-200 bg-red-50 rounded-lg p-6 text-center">
                    <X className="h-12 w-12 text-red-500 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-red-800 mb-2">Upload Failed</h3>
                    <p className="text-red-700 mb-4">
                      There was an error processing your file. Please try again.
                    </p>
                    <Button variant="outline" onClick={resetUpload}>
                      Try Again
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
          
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Tips</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-4">
                  <li className="flex">
                    <div className="mr-3 mt-0.5 bg-blue-100 p-1 rounded-full">
                      <Check className="h-4 w-4 text-blue-600" />
                    </div>
                    <p className="text-gray-700 text-sm">Ensure the report is clearly visible and all text is readable</p>
                  </li>
                  <li className="flex">
                    <div className="mr-3 mt-0.5 bg-blue-100 p-1 rounded-full">
                      <Check className="h-4 w-4 text-blue-600" />
                    </div>
                    <p className="text-gray-700 text-sm">Include all pages of multi-page reports</p>
                  </li>
                  <li className="flex">
                    <div className="mr-3 mt-0.5 bg-blue-100 p-1 rounded-full">
                      <Check className="h-4 w-4 text-blue-600" />
                    </div>
                    <p className="text-gray-700 text-sm">Make sure the report contains date and your identification</p>
                  </li>
                  <li className="flex">
                    <div className="mr-3 mt-0.5 bg-blue-100 p-1 rounded-full">
                      <Check className="h-4 w-4 text-blue-600" />
                    </div>
                    <p className="text-gray-700 text-sm">For better results, upload the original digital file rather than a photo</p>
                  </li>
                </ul>
                
                <div className="mt-6 pt-6 border-t border-gray-100">
                  <h4 className="font-medium text-gray-900 mb-2">Privacy Assurance</h4>
                  <p className="text-sm text-gray-700">
                    Your medical data is encrypted and securely processed. We never share your information with third parties without your explicit consent.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </motion.div>
    </Layout>
  );
};