import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  FileUp, 
  Pill, 
  ShoppingBag, 
  Bell, 
  BarChart, 
  Calendar, 
  HeartPulse,
  TrendingUp,
  Clock
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { useAuth } from '../../contexts/AuthContext';
import { mockHealthReports, mockPrescriptions, mockReminders, mockHealthMetrics, mockOrders } from '../../utils/mockData';

export const PatientDashboard: React.FC = () => {
  const { user } = useAuth();
  
  // Animation variants
  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.5 }
    }
  };
  
  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="pb-12">
      {/* Welcome Section */}
      <section className="bg-blue-700 text-white p-6 rounded-lg mb-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <h1 className="text-2xl font-bold mb-2">Welcome back, {user?.name}!</h1>
            <p className="text-blue-100">
              Here's an overview of your health management dashboard
            </p>
          </motion.div>
          
          <motion.div 
            initial="hidden"
            animate="visible"
            variants={fadeIn}
            className="mt-4 md:mt-0 flex space-x-2"
          >
            <Link to="/upload">
              <Button variant="secondary" size="sm" icon={<FileUp className="h-4 w-4" />}>
                Upload Report
              </Button>
            </Link>
            <Link to="/validator">
              <Button variant="outline" size="sm" className="bg-transparent border-white text-white hover:bg-white hover:text-blue-700" icon={<Pill className="h-4 w-4" />}>
                Validate Prescription
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
      
      {/* Quick Stats */}
      <motion.section 
        initial="hidden"
        animate="visible"
        variants={staggerContainer}
        className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
      >
        <motion.div variants={fadeIn}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="mr-4 bg-blue-100 p-3 rounded-full">
                  <HeartPulse className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Health Score</p>
                  <p className="text-2xl font-bold text-gray-900">86/100</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={fadeIn}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="mr-4 bg-green-100 p-3 rounded-full">
                  <Pill className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Active Medications</p>
                  <p className="text-2xl font-bold text-gray-900">2</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={fadeIn}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="mr-4 bg-amber-100 p-3 rounded-full">
                  <Bell className="h-6 w-6 text-amber-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Today's Reminders</p>
                  <p className="text-2xl font-bold text-gray-900">2</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={fadeIn}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="mr-4 bg-purple-100 p-3 rounded-full">
                  <ShoppingBag className="h-6 w-6 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Pending Orders</p>
                  <p className="text-2xl font-bold text-gray-900">1</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.section>
      
      {/* Main Dashboard Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-8">
          {/* Health Reports */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Recent Health Reports</CardTitle>
                <Link to="/reports">
                  <Button variant="outline" size="sm">View All</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockHealthReports.map(report => (
                    <div key={report.id} className="border-b border-gray-100 pb-4 last:border-0 last:pb-0">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h4 className="font-medium text-gray-900">
                            {report.originalFile}
                          </h4>
                          <p className="text-sm text-gray-500">{formatDate(report.date)}</p>
                        </div>
                        <Link to={`/report/${report.id}`}>
                          <Button variant="ghost" size="sm">View</Button>
                        </Link>
                      </div>
                      <p className="text-gray-700 text-sm line-clamp-2">{report.summary}</p>
                      <div className="flex mt-2 flex-wrap gap-2">
                        {report.conditions.map(condition => (
                          <span key={condition} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            {condition}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Health Metrics Chart */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Health Metrics</CardTitle>
                <div className="flex space-x-2">
                  <Button variant="ghost" size="sm">Week</Button>
                  <Button variant="ghost" size="sm">Month</Button>
                  <Button variant="ghost" size="sm" className="bg-blue-50 text-blue-700">Year</Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center">
                  <div className="w-full h-full flex flex-col">
                    <div className="flex justify-between mb-4">
                      <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div>
                        <span className="text-xs text-gray-600">Blood Pressure</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                        <span className="text-xs text-gray-600">Blood Sugar</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-purple-500 mr-2"></div>
                        <span className="text-xs text-gray-600">Weight</span>
                      </div>
                    </div>
                    
                    {/* Placeholder for chart - in a real app, use a chart library */}
                    <div className="flex-1 bg-gray-50 rounded-lg relative">
                      {/* Blue line (BP) */}
                      <div className="absolute top-1/4 left-0 right-0 h-0.5 bg-blue-500 opacity-70"></div>
                      <div className="absolute top-1/3 left-10 right-10 h-0.5 bg-blue-500 opacity-70" style={{transform: 'rotate(-2deg)'}}></div>
                      <div className="absolute top-1/5 left-20 right-0 h-0.5 bg-blue-500 opacity-70" style={{transform: 'rotate(1deg)'}}></div>
                      
                      {/* Green line (BS) */}
                      <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-green-500 opacity-70"></div>
                      <div className="absolute top-3/5 left-10 right-10 h-0.5 bg-green-500 opacity-70" style={{transform: 'rotate(3deg)'}}></div>
                      <div className="absolute top-2/4 left-20 right-0 h-0.5 bg-green-500 opacity-70" style={{transform: 'rotate(-2deg)'}}></div>
                      
                      {/* Purple line (Weight) */}
                      <div className="absolute top-3/4 left-0 right-0 h-0.5 bg-purple-500 opacity-70"></div>
                      <div className="absolute top-2/3 left-10 right-10 h-0.5 bg-purple-500 opacity-70" style={{transform: 'rotate(-1deg)'}}></div>
                      <div className="absolute top-4/5 left-20 right-0 h-0.5 bg-purple-500 opacity-70" style={{transform: 'rotate(1deg)'}}></div>
                      
                      <div className="absolute inset-0 flex items-center justify-center">
                        <p className="text-gray-400 text-sm">Interactive chart would be displayed here</p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-4 flex justify-center">
                  <Link to="/tracker">
                    <Button variant="outline" size="sm" icon={<BarChart className="h-4 w-4 mr-2" />}>
                      View Detailed Analytics
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Prescriptions */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Current Prescriptions</CardTitle>
                <Link to="/prescriptions">
                  <Button variant="outline" size="sm">View All</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockPrescriptions.map(prescription => (
                    <div key={prescription.id} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h4 className="font-medium text-gray-900">Dr. {prescription.doctorName}</h4>
                          <p className="text-sm text-gray-500">{formatDate(prescription.date)}</p>
                        </div>
                        <div className="flex items-center">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            prescription.validationStatus === 'validated' 
                              ? 'bg-green-100 text-green-800' 
                              : prescription.validationStatus === 'warning'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {prescription.validationStatus === 'validated' ? 'Validated' : 
                             prescription.validationStatus === 'warning' ? 'Warning' : 'Rejected'}
                          </span>
                        </div>
                      </div>
                      
                      <h5 className="text-sm font-medium text-gray-700 mb-2">Prescribed Medications:</h5>
                      <ul className="space-y-2">
                        {prescription.medicines.map(medicine => (
                          <li key={medicine.medicineId} className="flex justify-between">
                            <div>
                              <span className="text-gray-800">{medicine.medicineName}</span>
                              <p className="text-xs text-gray-500">
                                {medicine.dosage} - {medicine.frequency} for {medicine.duration}
                              </p>
                            </div>
                            <div className="flex space-x-2">
                              <Button variant="ghost" size="sm">Buy</Button>
                              <Button variant="ghost" size="sm">Remind</Button>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.section>
        </div>
        
        {/* Right Column */}
        <div className="space-y-8">
          {/* User Profile Card */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex flex-col items-center text-center">
                  {user?.profileImage ? (
                    <img 
                      src={user.profileImage} 
                      alt={user.name} 
                      className="h-24 w-24 rounded-full object-cover mb-4"
                    />
                  ) : (
                    <div className="h-24 w-24 rounded-full bg-blue-500 flex items-center justify-center text-white text-2xl font-bold mb-4">
                      {user?.name.charAt(0)}
                    </div>
                  )}
                  <h3 className="text-xl font-bold">{user?.name}</h3>
                  <p className="text-gray-500 mb-4">{user?.email}</p>
                  
                  <div className="w-full flex flex-col space-y-2">
                    <Link to="/profile">
                      <Button variant="outline" fullWidth>View Profile</Button>
                    </Link>
                    <Link to="/medical-history">
                      <Button variant="outline" fullWidth>Medical History</Button>
                    </Link>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Reminders */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Medication Reminders</CardTitle>
                <Link to="/reminders">
                  <Button variant="outline" size="sm">Manage</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {mockReminders.map(reminder => (
                    <div key={reminder.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center">
                        <div className="bg-blue-100 p-2 rounded-full mr-3">
                          <Clock className="h-5 w-5 text-blue-600" />
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900">{reminder.medicineName}</h4>
                          <p className="text-xs text-gray-500">{reminder.time} - {reminder.frequency}</p>
                        </div>
                      </div>
                      <div className="flex items-center">
                        <button className="p-1 hover:bg-gray-200 rounded-full">
                          <Bell className="h-4 w-4 text-gray-500" />
                        </button>
                      </div>
                    </div>
                  ))}
                  
                  <Button variant="ghost" fullWidth icon={<Calendar className="h-4 w-4 mr-2" />}>
                    Add New Reminder
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Orders */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Recent Orders</CardTitle>
                <Link to="/orders">
                  <Button variant="outline" size="sm">View All</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {mockOrders.map(order => (
                    <div key={order.id} className="border rounded-lg p-3">
                      <div className="flex justify-between mb-2">
                        <p className="text-sm font-medium">Order #{order.id.substring(0, 6)}</p>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          order.status === 'delivered' 
                            ? 'bg-green-100 text-green-800' 
                            : order.status === 'processing' || order.status === 'shipped'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {order.status.charAt(0).toUpperCase() + order.status.slice(1)}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mb-2">{formatDate(order.date)}</p>
                      <p className="text-sm text-gray-700 mb-1">
                        {order.items.length} items - ${order.totalAmount.toFixed(2)}
                      </p>
                      <Link to={`/order/${order.id}`}>
                        <Button variant="ghost" size="sm" icon={<TrendingUp className="h-4 w-4 mr-2" />}>
                          Track Order
                        </Button>
                      </Link>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.section>
        </div>
      </div>
    </div>
  );
};