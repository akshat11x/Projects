import React, { useState } from 'react';
import { Upload, Languages, FileText, Download } from 'lucide-react';

const ReportTranslator: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetLanguage, setTargetLanguage] = useState('es');
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedText, setTranslatedText] = useState<string | null>(null);

  const languages = [
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ru', name: 'Russian' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'hi', name: 'Hindi' }
  ];

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setTranslatedText(null);
    }
  };

  const handleTranslate = () => {
    if (!selectedFile) return;

    setIsTranslating(true);
    // Simulate translation process
    setTimeout(() => {
      setTranslatedText(`
        [Translated Medical Report]
        
        Patient Information:
        Name: John Doe
        Date of Birth: 01/15/1980
        
        Clinical Findings:
        - Blood pressure: 120/80 mmHg
        - Heart rate: 72 bpm
        - Temperature: 37°C
        
        Diagnosis:
        The patient presents normal vital signs.
        No significant abnormalities detected.
        
        Recommendations:
        1. Continue current medication
        2. Follow-up visit in 3 months
        3. Maintain healthy lifestyle
        
        Dr. Sarah Johnson
        Medical License #12345
      `);
      setIsTranslating(false);
    }, 2000);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Medical Report Translator</h1>

      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="p-6">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload Medical Report
            </label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg">
              <div className="space-y-1 text-center">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <div className="flex text-sm text-gray-600">
                  <label className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500">
                    <span>Upload a file</span>
                    <input
                      type="file"
                      className="sr-only"
                      accept=".pdf,.jpg,.jpeg,.png"
                      onChange={handleFileUpload}
                    />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">
                  PDF, PNG, JPG up to 10MB
                </p>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Language
            </label>
            <select
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            >
              {languages.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleTranslate}
            disabled={!selectedFile || isTranslating}
            className={`w-full flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              !selectedFile || isTranslating
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isTranslating ? (
              <>
                <Languages className="animate-spin -ml-1 mr-2 h-5 w-5" />
                Translating...
              </>
            ) : (
              <>
                <Languages className="-ml-1 mr-2 h-5 w-5" />
                Translate Report
              </>
            )}
          </button>
        </div>

        {translatedText && (
          <div className="border-t border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <FileText className="h-5 w-5 text-gray-400" />
                <h2 className="ml-2 text-lg font-medium text-gray-900">
                  Translated Report
                </h2>
              </div>
              <button className="flex items-center text-sm text-blue-600 hover:text-blue-500">
                <Download className="h-4 w-4 mr-1" />
                Download
              </button>
            </div>
            <pre className="bg-gray-50 rounded-lg p-4 text-sm text-gray-700 whitespace-pre-wrap">
              {translatedText}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportTranslator;