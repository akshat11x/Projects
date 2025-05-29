export interface User {
  id: string;
  name: string;
  email: string;
  role: 'patient' | 'pharmacist' | 'admin';
  profileImage?: string;
}

export interface HealthReport {
  id: string;
  userId: string;
  date: string;
  summary: string;
  originalFile: string;
  conditions: string[];
}

export interface Medicine {
  id: string;
  name: string;
  manufacturer: string;
  description: string;
  price: number;
  dosage: string;
  sideEffects: string[];
  imageUrl: string;
  category: string;
  inStock: boolean;
}

export interface Prescription {
  id: string;
  userId: string;
  date: string;
  medicines: {
    medicineId: string;
    medicineName: string;
    dosage: string;
    frequency: string;
    duration: string;
  }[];
  doctorName: string;
  validationStatus: 'pending' | 'validated' | 'warning' | 'rejected';
  notes?: string;
}

export interface Reminder {
  id: string;
  userId: string;
  medicineId: string;
  medicineName: string;
  time: string;
  frequency: 'daily' | 'twice-daily' | 'weekly' | 'monthly';
  notificationType: ('email' | 'sms' | 'push')[];
  active: boolean;
}

export interface HealthMetric {
  id: string;
  userId: string;
  type: 'blood-pressure' | 'blood-sugar' | 'weight' | 'symptoms';
  value: string | number;
  date: string;
  notes?: string;
}

export interface Order {
  id: string;
  userId: string;
  date: string;
  status: 'pending' | 'processing' | 'shipped' | 'delivered' | 'cancelled';
  items: {
    medicineId: string;
    medicineName: string;
    quantity: number;
    price: number;
  }[];
  totalAmount: number;
  shippingAddress: string;
  paymentMethod: string;
  trackingInfo?: string;
}