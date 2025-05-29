import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { FileUp, Pill, ShoppingBag, AlertCircle, Upload, Search, Clock, HeartPulse } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Card, CardContent } from '../components/ui/Card';

export const LandingPage: React.FC = () => {
  // Animation variants
  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.6 }
    }
  };
  
  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-blue-700 to-blue-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-28">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial="hidden"
              animate="visible"
              variants={fadeIn}
            >
              <h1 className="text-4xl md:text-5xl font-bold leading-tight mb-6">
                Your Complete Health Management Solution
              </h1>
              <p className="text-xl text-blue-100 mb-8">
                MED-KIT helps you understand your health reports, validate prescriptions, and manage your medicines with AI-powered insights.
              </p>
              <div className="flex flex-wrap gap-4">
                <Link to="/signup">
                  <Button size="lg" variant="secondary">
                    Get Started
                  </Button>
                </Link>
                <Link to="/upload">
                  <Button size="lg" variant="outline" className="bg-transparent border-white text-white hover:bg-white hover:text-blue-900">
                    Upload Report
                  </Button>
                </Link>
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8 }}
              className="hidden md:block"
            >
              <img 
                src="https://images.pexels.com/photos/7579831/pexels-photo-7579831.jpeg?auto=compress&cs=tinysrgb&w=600" 
                alt="Doctor using digital health technology" 
                className="rounded-lg shadow-2xl"
              />
            </motion.div>
          </div>
        </div>
      </section>
      
      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <motion.h2 
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeIn}
              className="text-3xl font-bold text-gray-900 mb-4"
            >
              Powerful Features for Your Health
            </motion.h2>
            <motion.p 
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeIn}
              className="text-xl text-gray-600 max-w-3xl mx-auto"
            >
              MED-KIT combines AI technology with healthcare expertise to provide you with tools that make health management simple and effective.
            </motion.p>
          </div>
          
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"
          >
            <motion.div variants={fadeIn}>
              <Card isHoverable className="h-full">
                <CardContent className="p-6 flex flex-col items-center text-center">
                  <div className="bg-blue-100 p-3 rounded-full mb-4">
                    <FileUp className="h-8 w-8 text-blue-700" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Health Report Analysis</h3>
                  <p className="text-gray-600">
                    Upload your medical reports and get simplified explanations and actionable insights.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div variants={fadeIn}>
              <Card isHoverable className="h-full">
                <CardContent className="p-6 flex flex-col items-center text-center">
                  <div className="bg-green-100 p-3 rounded-full mb-4">
                    <Pill className="h-8 w-8 text-green-700" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Prescription Validation</h3>
                  <p className="text-gray-600">
                    Verify prescriptions for potential drug interactions and side effects before taking medications.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div variants={fadeIn}>
              <Card isHoverable className="h-full">
                <CardContent className="p-6 flex flex-col items-center text-center">
                  <div className="bg-purple-100 p-3 rounded-full mb-4">
                    <ShoppingBag className="h-8 w-8 text-purple-700" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Medicine E-Commerce</h3>
                  <p className="text-gray-600">
                    Order verified medications directly through our platform with fast delivery to your doorstep.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div variants={fadeIn}>
              <Card isHoverable className="h-full">
                <CardContent className="p-6 flex flex-col items-center text-center">
                  <div className="bg-red-100 p-3 rounded-full mb-4">
                    <AlertCircle className="h-8 w-8 text-red-700" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Medicine Authentication</h3>
                  <p className="text-gray-600">
                    Scan medicine QR codes to verify authenticity and avoid counterfeit products.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* How It Works Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <motion.h2 
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeIn}
              className="text-3xl font-bold text-gray-900 mb-4"
            >
              How MED-KIT Works
            </motion.h2>
            <motion.p 
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeIn}
              className="text-xl text-gray-600 max-w-3xl mx-auto"
            >
              Our simple 4-step process makes health management effortless
            </motion.p>
          </div>
          
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="relative"
          >
            {/* Timeline line */}
            <div className="hidden md:block absolute left-1/2 transform -translate-x-1/2 h-full w-1 bg-blue-200"></div>
            
            <div className="space-y-12 md:space-y-0">
              {/* Step 1 */}
              <div className="md:grid md:grid-cols-2 md:gap-8 md:items-center">
                <motion.div 
                  variants={fadeIn}
                  className="md:text-right md:pr-12 mb-8 md:mb-0"
                >
                  <div className="bg-white p-6 rounded-lg shadow-md inline-block">
                    <h3 className="text-xl font-semibold text-blue-700 mb-2">Upload Your Reports</h3>
                    <p className="text-gray-600">
                      Simply scan or upload your medical reports to our secure platform. We accept PDF, images and various document formats.
                    </p>
                  </div>
                </motion.div>
                <motion.div 
                  variants={fadeIn}
                  className="hidden md:flex md:justify-start md:pl-12 items-center"
                >
                  <div className="bg-blue-600 p-4 rounded-full z-10">
                    <Upload className="h-8 w-8 text-white" />
                  </div>
                </motion.div>
              </div>
              
              {/* Step 2 */}
              <div className="md:grid md:grid-cols-2 md:gap-8 md:items-center">
                <motion.div 
                  variants={fadeIn}
                  className="md:order-2 md:text-left md:pl-12 mb-8 md:mb-0"
                >
                  <div className="bg-white p-6 rounded-lg shadow-md inline-block">
                    <h3 className="text-xl font-semibold text-blue-700 mb-2">AI Analysis</h3>
                    <p className="text-gray-600">
                      Our AI processes the reports, extracts key information, and provides a simplified summary of your health data.
                    </p>
                  </div>
                </motion.div>
                <motion.div 
                  variants={fadeIn}
                  className="hidden md:flex md:justify-end md:order-1 md:pr-12 items-center"
                >
                  <div className="bg-blue-600 p-4 rounded-full z-10">
                    <Search className="h-8 w-8 text-white" />
                  </div>
                </motion.div>
              </div>
              
              {/* Step 3 */}
              <div className="md:grid md:grid-cols-2 md:gap-8 md:items-center">
                <motion.div 
                  variants={fadeIn}
                  className="md:text-right md:pr-12 mb-8 md:mb-0"
                >
                  <div className="bg-white p-6 rounded-lg shadow-md inline-block">
                    <h3 className="text-xl font-semibold text-blue-700 mb-2">Set Up Reminders</h3>
                    <p className="text-gray-600">
                      Create medication reminders based on prescriptions. Get notifications via email, SMS, or app alerts.
                    </p>
                  </div>
                </motion.div>
                <motion.div 
                  variants={fadeIn}
                  className="hidden md:flex md:justify-start md:pl-12 items-center"
                >
                  <div className="bg-blue-600 p-4 rounded-full z-10">
                    <Clock className="h-8 w-8 text-white" />
                  </div>
                </motion.div>
              </div>
              
              {/* Step 4 */}
              <div className="md:grid md:grid-cols-2 md:gap-8 md:items-center">
                <motion.div 
                  variants={fadeIn}
                  className="md:order-2 md:text-left md:pl-12"
                >
                  <div className="bg-white p-6 rounded-lg shadow-md inline-block">
                    <h3 className="text-xl font-semibold text-blue-700 mb-2">Track Your Health</h3>
                    <p className="text-gray-600">
                      Monitor your health metrics over time, track improvements, and get insights into your overall wellbeing.
                    </p>
                  </div>
                </motion.div>
                <motion.div 
                  variants={fadeIn}
                  className="hidden md:flex md:justify-end md:order-1 md:pr-12 items-center"
                >
                  <div className="bg-blue-600 p-4 rounded-full z-10">
                    <HeartPulse className="h-8 w-8 text-white" />
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      
      {/* Testimonials Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <motion.h2 
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeIn}
              className="text-3xl font-bold text-gray-900 mb-4"
            >
              What Our Users Say
            </motion.h2>
            <motion.p 
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeIn}
              className="text-xl text-gray-600 max-w-3xl mx-auto"
            >
              Join thousands of satisfied users who have transformed their health management
            </motion.p>
          </div>
          
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            <motion.div variants={fadeIn}>
              <Card className="h-full">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <div className="mr-4">
                      <img 
                        src="https://images.pexels.com/photos/1222271/pexels-photo-1222271.jpeg?auto=compress&cs=tinysrgb&w=100" 
                        alt="User" 
                        className="h-12 w-12 rounded-full object-cover"
                      />
                    </div>
                    <div>
                      <h4 className="font-medium">David Johnson</h4>
                      <p className="text-gray-500 text-sm">Diabetic Patient</p>
                    </div>
                  </div>
                  <p className="text-gray-600 italic">
                    "MED-KIT has transformed how I manage my diabetes. The simplified report summaries help me understand my condition better, and the medication reminders ensure I never miss a dose."
                  </p>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div variants={fadeIn}>
              <Card className="h-full">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <div className="mr-4">
                      <img 
                        src="https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=100" 
                        alt="User" 
                        className="h-12 w-12 rounded-full object-cover"
                      />
                    </div>
                    <div>
                      <h4 className="font-medium">Sarah Williams</h4>
                      <p className="text-gray-500 text-sm">Caregiver</p>
                    </div>
                  </div>
                  <p className="text-gray-600 italic">
                    "As someone who cares for an elderly parent, MED-KIT has been invaluable. The prescription validation feature gives me peace of mind, and the medicine authentication ensures we're getting genuine products."
                  </p>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div variants={fadeIn}>
              <Card className="h-full">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <div className="mr-4">
                      <img 
                        src="https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg?auto=compress&cs=tinysrgb&w=100" 
                        alt="User" 
                        className="h-12 w-12 rounded-full object-cover"
                      />
                    </div>
                    <div>
                      <h4 className="font-medium">Dr. Michael Chen</h4>
                      <p className="text-gray-500 text-sm">Pharmacist</p>
                    </div>
                  </div>
                  <p className="text-gray-600 italic">
                    "From a healthcare professional's perspective, MED-KIT is an excellent tool. The platform helps patients better understand their medications and adhere to treatment plans, which ultimately leads to better outcomes."
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-20 bg-blue-700 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.h2 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeIn}
            className="text-3xl font-bold mb-6"
          >
            Ready to Take Control of Your Health?
          </motion.h2>
          <motion.p 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeIn}
            className="text-xl text-blue-100 mb-8 max-w-3xl mx-auto"
          >
            Join MED-KIT today and experience a smarter way to manage your health. Sign up for free and get started in minutes.
          </motion.p>
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeIn}
          >
            <Link to="/signup">
              <Button size="lg" variant="secondary" className="mr-4">
                Sign Up Now
              </Button>
            </Link>
            <Link to="/about">
              <Button size="lg" variant="outline" className="bg-transparent border-white text-white hover:bg-white hover:text-blue-700">
                Learn More
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
      
      {/* Trust & Safety Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.h3 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeIn}
            className="text-2xl font-semibold text-gray-800 mb-8"
          >
            Your Health Data is Safe With Us
          </motion.h3>
          <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="flex flex-wrap justify-center gap-8"
          >
            <motion.div variants={fadeIn} className="flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-2">
                <svg className="h-5 w-5 text-blue-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <span className="text-gray-700">HIPAA Compliant</span>
            </motion.div>
            <motion.div variants={fadeIn} className="flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-2">
                <svg className="h-5 w-5 text-blue-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <span className="text-gray-700">256-bit Encryption</span>
            </motion.div>
            <motion.div variants={fadeIn} className="flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-2">
                <svg className="h-5 w-5 text-blue-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <span className="text-gray-700">Data Privacy Guaranteed</span>
            </motion.div>
            <motion.div variants={fadeIn} className="flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-2">
                <svg className="h-5 w-5 text-blue-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <span className="text-gray-700">FDA Registered</span>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};