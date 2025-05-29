import { User, HealthReport, Medicine, Prescription, Reminder, HealthMetric, Order } from '../types';

export const mockUser: User = {
  id: 'user1',
  name: 'Jane Smith',
  email: 'jane.smith@example.com',
  role: 'patient',
  profileImage: 'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg?auto=compress&cs=tinysrgb&w=150'
};

export const mockHealthReports: HealthReport[] = [
  {
    id: 'report1',
    userId: 'user1',
    date: '2024-04-15',
    summary: 'Complete blood count shows normal results. Cholesterol levels are slightly elevated. Recommended to reduce saturated fat intake and increase exercise.',
    originalFile: 'health-report-2024-04.pdf',
    conditions: ['Mild hypercholesterolemia']
  },
  {
    id: 'report2',
    userId: 'user1',
    date: '2024-01-10',
    summary: 'Annual checkup results show normal vitals. Vitamin D levels are below optimal range. Recommended supplementation.',
    originalFile: 'annual-checkup-2024.pdf',
    conditions: ['Vitamin D deficiency']
  }
];

export const mockMedicines: Medicine[] = [
  {
    id: 'med1',
    name: 'Lipilow',
    manufacturer: 'MediPharma',
    description: 'Cholesterol reduction medication',
    price: 32.99,
    dosage: '10mg',
    sideEffects: ['Muscle pain', 'Headache', 'Nausea'],
    imageUrl: 'https://images.pexels.com/photos/139398/pexels-photo-139398.jpeg?auto=compress&cs=tinysrgb&w=300',
    category: 'Cardiovascular',
    inStock: true
  },
  {
    id: 'med2',
    name: 'VitaD3',
    manufacturer: 'NutriHealth',
    description: 'Vitamin D3 supplement',
    price: 15.49,
    dosage: '1000 IU',
    sideEffects: ['Rare allergic reactions'],
    imageUrl: 'https://images.pexels.com/photos/3683074/pexels-photo-3683074.jpeg?auto=compress&cs=tinysrgb&w=300',
    category: 'Supplements',
    inStock: true
  },
  {
    id: 'med3',
    name: 'Paracetol',
    manufacturer: 'ReliefMed',
    description: 'Pain reliever and fever reducer',
    price: 8.99,
    dosage: '500mg',
    sideEffects: ['Liver damage (with overuse)', 'Nausea'],
    imageUrl: 'https://images.pexels.com/photos/159211/headache-pain-pills-medication-159211.jpeg?auto=compress&cs=tinysrgb&w=300',
    category: 'Pain Relief',
    inStock: true
  },
  {
    id: 'med4',
    name: 'Insulite',
    manufacturer: 'DiaCare',
    description: 'Diabetes management medication',
    price: 45.99,
    dosage: '5mg',
    sideEffects: ['Hypoglycemia', 'Weight gain', 'Dizziness'],
    imageUrl: 'https://images.pexels.com/photos/4210611/pexels-photo-4210611.jpeg?auto=compress&cs=tinysrgb&w=300',
    category: 'Diabetes',
    inStock: false
  }
];

export const mockPrescriptions: Prescription[] = [
  {
    id: 'presc1',
    userId: 'user1',
    date: '2024-04-16',
    medicines: [
      {
        medicineId: 'med1',
        medicineName: 'Lipilow',
        dosage: '10mg',
        frequency: 'Once daily',
        duration: '3 months'
      },
      {
        medicineId: 'med2',
        medicineName: 'VitaD3',
        dosage: '1000 IU',
        frequency: 'Once daily',
        duration: '6 months'
      }
    ],
    doctorName: 'Dr. Robert Chen',
    validationStatus: 'validated'
  }
];

export const mockReminders: Reminder[] = [
  {
    id: 'rem1',
    userId: 'user1',
    medicineId: 'med1',
    medicineName: 'Lipilow',
    time: '08:00',
    frequency: 'daily',
    notificationType: ['push', 'email'],
    active: true
  },
  {
    id: 'rem2',
    userId: 'user1',
    medicineId: 'med2',
    medicineName: 'VitaD3',
    time: '09:00',
    frequency: 'daily',
    notificationType: ['push'],
    active: true
  }
];

export const mockHealthMetrics: HealthMetric[] = [
  {
    id: 'metric1',
    userId: 'user1',
    type: 'blood-pressure',
    value: '120/80',
    date: '2024-04-16',
    notes: 'Morning reading'
  },
  {
    id: 'metric2',
    userId: 'user1',
    type: 'blood-pressure',
    value: '118/78',
    date: '2024-04-15',
    notes: 'Morning reading'
  },
  {
    id: 'metric3',
    userId: 'user1',
    type: 'blood-sugar',
    value: 95,
    date: '2024-04-16',
    notes: 'Fasting'
  },
  {
    id: 'metric4',
    userId: 'user1',
    type: 'blood-sugar',
    value: 92,
    date: '2024-04-15',
    notes: 'Fasting'
  },
  {
    id: 'metric5',
    userId: 'user1',
    type: 'weight',
    value: 68.5,
    date: '2024-04-16'
  },
  {
    id: 'metric6',
    userId: 'user1',
    type: 'weight',
    value: 68.8,
    date: '2024-04-13'
  }
];

export const mockOrders: Order[] = [
  {
    id: 'order1',
    userId: 'user1',
    date: '2024-04-16',
    status: 'processing',
    items: [
      {
        medicineId: 'med1',
        medicineName: 'Lipilow',
        quantity: 1,
        price: 32.99
      },
      {
        medicineId: 'med2',
        medicineName: 'VitaD3',
        quantity: 2,
        price: 15.49
      }
    ],
    totalAmount: 63.97,
    shippingAddress: '123 Health St, Medical City, MC 12345',
    paymentMethod: 'Credit Card'
  }
];