import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Layout } from '../components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { AlertCircle, Check, X, Search, Plus, Eye, Camera, Upload } from 'lucide-react';

export const MedicineValidatorPage: React.FC = () => {
  const [medicineName, setMedicineName] = useState('');
  const [validationStatus, setValidationStatus] = useState<'idle' | 'validating' | 'safe' | 'warning' | 'danger'>('idle');
  const [validationResults, setValidationResults] = useState<{
    medicine: string;
    dosage: string;
    sideEffects: string[];
    interactions: string[];
    recommendations: string;
    riskLevel: 'safe' | 'warning' | 'danger';
  } | null>(null);
  
  const handleValidate = () => {
    if (!medicineName) return;
    
    setValidationStatus('validating');
    
    // Simulate validation process
    setTimeout(() => {
      if (medicineName.toLowerCase().includes('lipilow')) {
        setValidationStatus('safe');
        setValidationResults({
          medicine: 'Lipilow (Atorvastatin) 10mg',
          dosage: 'Standard dosage: 10-80mg once daily',
          sideEffects: [
            'Muscle pain or weakness',
            'Headache',
            'Digestive issues',
            'Mild liver enzyme elevation'
          ],
          interactions: [
            'Grapefruit juice may increase side effects',
            'Some antibiotics may increase risk of muscle damage',
            'No known interaction with your current medications'
          ],
          recommendations: 'Safe to use as prescribed. Take with or without food, preferably at the same time each day.',
          riskLevel: 'safe'
        });
      } else if (medicineName.toLowerCase().includes('amox')) {
        setValidationStatus('warning');
        setValidationResults({
          medicine: 'Amoxicillin 500mg',
          dosage: 'Standard dosage: 250-500mg three times daily',
          sideEffects: [
            'Diarrhea',
            'Stomach upset',
            'Rash',
            'Allergic reactions'
          ],
          interactions: [
            'May reduce effectiveness of birth control pills',
            'Potential interaction with blood thinners',
            'WARNING: Your profile indicates a mild penicillin allergy'
          ],
          recommendations: 'Use with caution. Consult your doctor before taking due to your history of mild penicillin allergy.',
          riskLevel: 'warning'
        });
      } else {
        setValidationStatus('danger');
        setValidationResults({
          medicine: medicineName,
          dosage: 'Unable to verify standard dosage',
          sideEffects: [
            'Unknown - medicine not found in our database',
            'Potential for adverse reactions'
          ],
          interactions: [
            'Cannot verify safety with your current medications',
            'DANGER: This medication is not recognized in our system'
          ],
          recommendations: 'DO NOT TAKE. This medication could not be verified. Please consult your doctor or pharmacist immediately.',
          riskLevel: 'danger'
        });
      }
    }, 2000);
  };

  const resetValidation = () => {
    setMedicineName('');
    setValidationStatus('idle');
    setValidationResults(null);
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
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Prescription Validator</h1>
        <p className="text-lg text-gray-600 mb-8">
          Verify the safety and potential side effects of your medications before taking them.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Validate Your Medication</CardTitle>
                <CardDescription>
                  Enter medication names, upload a prescription, or scan medicine packaging
                </CardDescription>
              </CardHeader>
              <CardContent>
                {validationStatus === 'idle' && (
                  <div className="space-y-6">
                    <div className="flex flex-col space-y-4">
                      <div className="relative">
                        <Input
                          placeholder="Enter medicine name..."
                          value={medicineName}
                          onChange={(e) => setMedicineName(e.target.value)}
                          icon={<Search className="h-5 w-5" />}
                          fullWidth
                        />
                        <Button 
                          variant="ghost" 
                          className="absolute right-2 top-1/2 transform -translate-y-1/2"
                          onClick={() => medicineName && handleValidate()}
                        >
                          Validate
                        </Button>
                      </div>
                      
                      <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
                        <Button variant="outline" icon={<Upload className="h-4 w-4 mr-2" />} fullWidth>
                          Upload Prescription
                        </Button>
                        <Button variant="outline" icon={<Camera className="h-4 w-4 mr-2" />} fullWidth>
                          Scan Medicine
                        </Button>
                      </div>
                    </div>
                    
                    <div className="border-t border-gray-200 pt-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4">Recently Validated</h3>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center">
                            <div className="mr-3 bg-green-100 p-1 rounded-full">
                              <Check className="h-4 w-4 text-green-600" />
                            </div>
                            <span className="text-gray-700">Vitamin D3 1000 IU</span>
                          </div>
                          <Button variant="ghost" size="sm" icon={<Eye className="h-4 w-4" />}>
                            View
                          </Button>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center">
                            <div className="mr-3 bg-yellow-100 p-1 rounded-full">
                              <AlertCircle className="h-4 w-4 text-yellow-600" />
                            </div>
                            <span className="text-gray-700">Ibuprofen 600mg</span>
                          </div>
                          <Button variant="ghost" size="sm" icon={<Eye className="h-4 w-4" />}>
                            View
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {validationStatus === 'validating' && (
                  <div className="py-12 text-center">
                    <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent align-[-0.125em]"></div>
                    <p className="mt-4 text-lg text-gray-700">Validating medication...</p>
                    <p className="text-sm text-gray-500">
                      Checking for interactions, side effects, and compatibility with your health profile
                    </p>
                  </div>
                )}
                
                {validationResults && (validationStatus === 'safe' || validationStatus === 'warning' || validationStatus === 'danger') && (
                  <div className="space-y-6">
                    <div className="flex items-center space-x-4">
                      <div className={`p-3 rounded-full ${
                        validationStatus === 'safe' ? 'bg-green-100' : 
                        validationStatus === 'warning' ? 'bg-yellow-100' : 'bg-red-100'
                      }`}>
                        {validationStatus === 'safe' ? (
                          <Check className={`h-6 w-6 text-green-600`} />
                        ) : validationStatus === 'warning' ? (
                          <AlertCircle className={`h-6 w-6 text-yellow-600`} />
                        ) : (
                          <X className={`h-6 w-6 text-red-600`} />
                        )}
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold">{validationResults.medicine}</h3>
                        <p className={`text-sm ${
                          validationStatus === 'safe' ? 'text-green-700' : 
                          validationStatus === 'warning' ? 'text-yellow-700' : 'text-red-700'
                        }`}>
                          {validationStatus === 'safe' ? 'Safe to use as prescribed' : 
                           validationStatus === 'warning' ? 'Use with caution' : 'Not recommended'}
                        </p>
                      </div>
                    </div>
                    
                    <div className={`p-4 rounded-lg border ${
                      validationStatus === 'safe' ? 'border-green-200 bg-green-50' : 
                      validationStatus === 'warning' ? 'border-yellow-200 bg-yellow-50' : 'border-red-200 bg-red-50'
                    }`}>
                      <p className={`font-medium ${
                        validationStatus === 'safe' ? 'text-green-700' : 
                        validationStatus === 'warning' ? 'text-yellow-700' : 'text-red-700'
                      }`}>
                        {validationResults.recommendations}
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Dosage Information</h4>
                        <p className="text-gray-700 text-sm">{validationResults.dosage}</p>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Common Side Effects</h4>
                        <ul className="text-sm text-gray-700 space-y-1">
                          {validationResults.sideEffects.map((effect, index) => (
                            <li key={index} className="flex items-start">
                              <span className="mr-2">•</span>
                              <span>{effect}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Potential Interactions</h4>
                      <ul className="text-sm text-gray-700 space-y-1">
                        {validationResults.interactions.map((interaction, index) => (
                          <li key={index} className="flex items-start">
                            <span className="mr-2">•</span>
                            <span 
                              className={interaction.includes('WARNING') || interaction.includes('DANGER') ? 
                                'font-medium text-red-600' : ''}
                            >
                              {interaction}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div className="flex space-x-3 pt-4 border-t border-gray-200">
                      <Button variant="outline" onClick={resetValidation}>
                        Validate Another
                      </Button>
                      <Button 
                        variant="primary" 
                        icon={<Plus className="h-4 w-4 mr-2" />}
                        disabled={validationStatus === 'danger'}
                      >
                        Add to My Medications
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
          
          <div>
            <Card>
              <CardHeader>
                <CardTitle>How It Works</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <div className="flex items-center mb-2">
                      <div className="bg-blue-100 p-1 rounded-full mr-2">
                        <span className="text-blue-700 font-medium text-xs">1</span>
                      </div>
                      <h4 className="font-medium text-gray-900">Enter Medicine Details</h4>
                    </div>
                    <p className="text-sm text-gray-700 ml-7">
                      Type the name of your medication or upload a prescription
                    </p>
                  </div>
                  
                  <div>
                    <div className="flex items-center mb-2">
                      <div className="bg-blue-100 p-1 rounded-full mr-2">
                        <span className="text-blue-700 font-medium text-xs">2</span>
                      </div>
                      <h4 className="font-medium text-gray-900">AI Analysis</h4>
                    </div>
                    <p className="text-sm text-gray-700 ml-7">
                      Our AI checks for potential risks, side effects, and interactions with your other medications
                    </p>
                  </div>
                  
                  <div>
                    <div className="flex items-center mb-2">
                      <div className="bg-blue-100 p-1 rounded-full mr-2">
                        <span className="text-blue-700 font-medium text-xs">3</span>
                      </div>
                      <h4 className="font-medium text-gray-900">Review Results</h4>
                    </div>
                    <p className="text-sm text-gray-700 ml-7">
                      Get a safety assessment and recommendations in simple language
                    </p>
                  </div>
                </div>
                
                <div className="mt-6 pt-6 border-t border-gray-100">
                  <h4 className="font-medium text-gray-900 mb-2">Important Note</h4>
                  <p className="text-sm text-gray-700">
                    This tool is not a replacement for professional medical advice. Always consult your doctor or pharmacist before starting, stopping, or changing any medication.
                  </p>
                </div>
              </CardContent>
            </Card>
            
            <div className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Health Profile</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-700 mb-4">
                    Keep your health profile updated for more accurate medication validation.
                  </p>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Allergies</span>
                      <span className="text-sm font-medium">Mild penicillin allergy</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Current Medications</span>
                      <span className="text-sm font-medium">2 medications</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Medical Conditions</span>
                      <span className="text-sm font-medium">Mild hypercholesterolemia</span>
                    </div>
                  </div>
                  
                  <Button variant="outline" size="sm" className="w-full mt-4">
                    Update Health Profile
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </motion.div>
    </Layout>
  );
};