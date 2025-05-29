import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  Package, 
  Users, 
  ShoppingBag, 
  AlertCircle, 
  TrendingUp,
  Pill,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  List
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { useAuth } from '../../contexts/AuthContext';
import { mockMedicines, mockOrders } from '../../utils/mockData';

export const PharmacistDashboard: React.FC = () => {
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

  const pendingPrescriptions = [
    {
      id: 'presc2',
      patientName: 'John Davis',
      date: '2024-04-18',
      medicine: 'Amoxicillin 500mg',
      priority: 'high'
    },
    {
      id: 'presc3',
      patientName: 'Emily Wilson',
      date: '2024-04-18',
      medicine: 'Metformin 1000mg',
      priority: 'medium'
    },
    {
      id: 'presc4',
      patientName: 'Michael Brown',
      date: '2024-04-17',
      medicine: 'Atorvastatin 40mg',
      priority: 'low'
    },
  ];

  const recentOrders = [
    {
      id: 'order2',
      patientName: 'Lisa Johnson',
      date: '2024-04-18',
      status: 'pending',
      items: 3,
      total: 72.45
    },
    {
      id: 'order3',
      patientName: 'Robert Wilson',
      date: '2024-04-17',
      status: 'processing',
      items: 1,
      total: 45.99
    },
    {
      id: 'order4',
      patientName: 'Sarah Miller',
      date: '2024-04-16',
      status: 'shipped',
      items: 2,
      total: 29.98
    },
  ];

  const lowStockItems = mockMedicines
    .filter(med => !med.inStock)
    .map(med => ({
      id: med.id,
      name: med.name,
      currentStock: 2,
      minimumRequired: 10,
      supplier: med.manufacturer
    }));

  return (
    <div className="pb-12">
      {/* Welcome Section */}
      <section className="bg-teal-700 text-white p-6 rounded-lg mb-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <h1 className="text-2xl font-bold mb-2">Welcome back, {user?.name}!</h1>
            <p className="text-teal-100">
              Pharmacist Dashboard - Manage inventory, orders, and prescriptions
            </p>
          </motion.div>
          
          <motion.div 
            initial="hidden"
            animate="visible"
            variants={fadeIn}
            className="mt-4 md:mt-0 flex space-x-2"
          >
            <Link to="/inventory">
              <Button variant="secondary" size="sm" icon={<Package className="h-4 w-4" />}>
                Manage Inventory
              </Button>
            </Link>
            <Link to="/orders/pending">
              <Button variant="outline" size="sm" className="bg-transparent border-white text-white hover:bg-white hover:text-teal-700" icon={<ShoppingBag className="h-4 w-4" />}>
                Process Orders
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
                <div className="mr-4 bg-teal-100 p-3 rounded-full">
                  <ShoppingBag className="h-6 w-6 text-teal-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Pending Orders</p>
                  <p className="text-2xl font-bold text-gray-900">12</p>
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
                  <FileText className="h-6 w-6 text-amber-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Prescriptions</p>
                  <p className="text-2xl font-bold text-gray-900">8</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={fadeIn}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="mr-4 bg-red-100 p-3 rounded-full">
                  <AlertCircle className="h-6 w-6 text-red-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Low Stock Items</p>
                  <p className="text-2xl font-bold text-gray-900">3</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={fadeIn}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="mr-4 bg-blue-100 p-3 rounded-full">
                  <Users className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Active Customers</p>
                  <p className="text-2xl font-bold text-gray-900">156</p>
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
          {/* Pending Prescriptions */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Pending Prescriptions</CardTitle>
                <Link to="/prescriptions/pending">
                  <Button variant="outline" size="sm">View All</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {pendingPrescriptions.map(prescription => (
                    <div key={prescription.id} className="border-b border-gray-100 pb-4 last:border-0 last:pb-0">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-start">
                          <div className={`mt-1 h-2 w-2 rounded-full mr-2 ${
                            prescription.priority === 'high' ? 'bg-red-500' :
                            prescription.priority === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                          }`}></div>
                          <div>
                            <h4 className="font-medium text-gray-900">
                              {prescription.patientName}
                            </h4>
                            <p className="text-sm text-gray-500">{formatDate(prescription.date)}</p>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <Button variant="outline" size="sm" icon={<CheckCircle className="h-4 w-4" />}>
                            Approve
                          </Button>
                          <Button variant="outline" size="sm" icon={<XCircle className="h-4 w-4" />}>
                            Reject
                          </Button>
                        </div>
                      </div>
                      <p className="text-gray-700 text-sm">{prescription.medicine}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Orders Overview */}
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
                <div className="space-y-4">
                  {recentOrders.map(order => (
                    <div key={order.id} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h4 className="font-medium text-gray-900">{order.patientName}</h4>
                          <p className="text-sm text-gray-500">Order #{order.id.substring(0, 6)} - {formatDate(order.date)}</p>
                        </div>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          order.status === 'shipped' 
                            ? 'bg-green-100 text-green-800' 
                            : order.status === 'processing'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {order.status.charAt(0).toUpperCase() + order.status.slice(1)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="text-sm text-gray-700">
                            {order.items} items - ${order.total.toFixed(2)}
                          </p>
                        </div>
                        <div className="flex space-x-2">
                          <Button variant="outline" size="sm">Process</Button>
                          <Button variant="ghost" size="sm">Details</Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Sales Chart */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Sales Overview</CardTitle>
                <div className="flex space-x-2">
                  <Button variant="ghost" size="sm">Week</Button>
                  <Button variant="ghost" size="sm" className="bg-teal-50 text-teal-700">Month</Button>
                  <Button variant="ghost" size="sm">Year</Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center">
                  <div className="w-full h-full flex flex-col">
                    <div className="flex justify-between mb-4">
                      <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-teal-500 mr-2"></div>
                        <span className="text-xs text-gray-600">Orders</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div>
                        <span className="text-xs text-gray-600">Revenue</span>
                      </div>
                    </div>
                    
                    {/* Placeholder for chart - in a real app, use a chart library */}
                    <div className="flex-1 bg-gray-50 rounded-lg relative">
                      {/* Teal bars (Orders) */}
                      <div className="absolute bottom-0 left-[10%] w-4 h-[30%] bg-teal-500 opacity-70 rounded-t"></div>
                      <div className="absolute bottom-0 left-[20%] w-4 h-[40%] bg-teal-500 opacity-70 rounded-t"></div>
                      <div className="absolute bottom-0 left-[30%] w-4 h-[25%] bg-teal-500 opacity-70 rounded-t"></div>
                      <div className="absolute bottom-0 left-[40%] w-4 h-[45%] bg-teal-500 opacity-70 rounded-t"></div>
                      <div className="absolute bottom-0 left-[50%] w-4 h-[60%] bg-teal-500 opacity-70 rounded-t"></div>
                      <div className="absolute bottom-0 left-[60%] w-4 h-[50%] bg-teal-500 opacity-70 rounded-t"></div>
                      <div className="absolute bottom-0 left-[70%] w-4 h-[70%] bg-teal-500 opacity-70 rounded-t"></div>
                      <div className="absolute bottom-0 left-[80%] w-4 h-[55%] bg-teal-500 opacity-70 rounded-t"></div>
                      
                      {/* Blue line (Revenue) */}
                      <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-blue-500 opacity-70"></div>
                      <div className="absolute top-3/5 left-10 right-10 h-0.5 bg-blue-500 opacity-70" style={{transform: 'rotate(-3deg)'}}></div>
                      <div className="absolute top-2/5 left-20 right-0 h-0.5 bg-blue-500 opacity-70" style={{transform: 'rotate(2deg)'}}></div>
                      
                      <div className="absolute inset-0 flex items-center justify-center">
                        <p className="text-gray-400 text-sm">Interactive sales chart would be displayed here</p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-4 flex justify-center">
                  <Link to="/analytics">
                    <Button variant="outline" size="sm" icon={<TrendingUp className="h-4 w-4 mr-2" />}>
                      View Detailed Analytics
                    </Button>
                  </Link>
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
                    <div className="h-24 w-24 rounded-full bg-teal-500 flex items-center justify-center text-white text-2xl font-bold mb-4">
                      {user?.name.charAt(0)}
                    </div>
                  )}
                  <h3 className="text-xl font-bold">{user?.name}</h3>
                  <p className="text-gray-500 mb-4">Pharmacist</p>
                  
                  <div className="w-full flex flex-col space-y-2">
                    <Link to="/profile">
                      <Button variant="outline" fullWidth>View Profile</Button>
                    </Link>
                    <Link to="/settings">
                      <Button variant="outline" fullWidth>Settings</Button>
                    </Link>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Upcoming Tasks */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Today's Tasks</CardTitle>
                <Link to="/tasks">
                  <Button variant="outline" size="sm">View All</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <div className="bg-blue-100 p-2 rounded-full mr-3">
                        <Clock className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">Process pending orders</h4>
                        <p className="text-xs text-gray-500">9:30 AM - High Priority</p>
                      </div>
                    </div>
                    <input type="checkbox" className="h-4 w-4 text-teal-600 rounded" />
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <div className="bg-blue-100 p-2 rounded-full mr-3">
                        <Pill className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">Update inventory</h4>
                        <p className="text-xs text-gray-500">11:00 AM - Medium Priority</p>
                      </div>
                    </div>
                    <input type="checkbox" className="h-4 w-4 text-teal-600 rounded" />
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <div className="bg-blue-100 p-2 rounded-full mr-3">
                        <List className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">Review prescriptions</h4>
                        <p className="text-xs text-gray-500">2:00 PM - High Priority</p>
                      </div>
                    </div>
                    <input type="checkbox" className="h-4 w-4 text-teal-600 rounded" />
                  </div>
                  
                  <Button variant="ghost" fullWidth>
                    Add New Task
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.section>
          
          {/* Low Stock Alert */}
          <motion.section
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Low Stock Alert</CardTitle>
                <Link to="/inventory/low-stock">
                  <Button variant="outline" size="sm">Manage</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {lowStockItems.map(item => (
                    <div key={item.id} className="border rounded-lg p-3">
                      <div className="flex justify-between mb-2">
                        <h4 className="font-medium text-gray-900">{item.name}</h4>
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          Low Stock
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-sm text-gray-600">
                        <span>Current: {item.currentStock}</span>
                        <span>Required: {item.minimumRequired}</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Supplier: {item.supplier}</p>
                      <Button variant="ghost" size="sm" className="mt-2 w-full">
                        Reorder
                      </Button>
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