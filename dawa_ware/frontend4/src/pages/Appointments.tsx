import React, { useState } from 'react';
import { Calendar, Clock, MapPin, User } from 'lucide-react';

interface Doctor {
  id: string;
  name: string;
  specialty: string;
  rating: number;
  image: string;
  availableSlots: string[];
}

const Appointments: React.FC = () => {
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedDoctor, setSelectedDoctor] = useState<string>('');
  const [selectedTime, setSelectedTime] = useState<string>('');

  const doctors: Doctor[] = [
    {
      id: '1',
      name: 'Dr. Sarah Johnson',
      specialty: 'Cardiologist',
      rating: 4.8,
      image: 'https://images.pexels.com/photos/5452293/pexels-photo-5452293.jpeg',
      availableSlots: ['09:00 AM', '10:00 AM', '2:00 PM', '3:00 PM']
    },
    {
      id: '2',
      name: 'Dr. Michael Chen',
      specialty: 'Neurologist',
      rating: 4.9,
      image: 'https://images.pexels.com/photos/5452201/pexels-photo-5452201.jpeg',
      availableSlots: ['11:00 AM', '1:00 PM', '4:00 PM']
    }
  ];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Book an Appointment</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Calendar Section */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Select Date</h2>
            <div className="grid grid-cols-7 gap-2">
              {Array.from({ length: 31 }, (_, i) => i + 1).map((day) => (
                <button
                  key={day}
                  onClick={() => setSelectedDate(`2024-03-${day}`)}
                  className={`p-2 text-center rounded-lg hover:bg-blue-50 ${
                    selectedDate === `2024-03-${day}`
                      ? 'bg-blue-100 text-blue-600'
                      : 'text-gray-700'
                  }`}
                >
                  {day}
                </button>
              ))}
            </div>
          </div>

          {/* Doctors List */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Select Doctor</h2>
            <div className="space-y-4">
              {doctors.map((doctor) => (
                <div
                  key={doctor.id}
                  className={`p-4 rounded-lg border cursor-pointer ${
                    selectedDoctor === doctor.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-blue-300'
                  }`}
                  onClick={() => setSelectedDoctor(doctor.id)}
                >
                  <div className="flex items-center space-x-4">
                    <img
                      src={doctor.image}
                      alt={doctor.name}
                      className="h-16 w-16 rounded-full object-cover"
                    />
                    <div>
                      <h3 className="font-medium text-gray-900">{doctor.name}</h3>
                      <p className="text-sm text-gray-500">{doctor.specialty}</p>
                      <div className="flex items-center mt-1">
                        <span className="text-yellow-400">★</span>
                        <span className="ml-1 text-sm text-gray-600">
                          {doctor.rating}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Time Slots */}
          {selectedDoctor && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Select Time</h2>
              <div className="grid grid-cols-3 gap-3">
                {doctors
                  .find((d) => d.id === selectedDoctor)
                  ?.availableSlots.map((slot) => (
                    <button
                      key={slot}
                      onClick={() => setSelectedTime(slot)}
                      className={`p-2 text-center rounded-lg border ${
                        selectedTime === slot
                          ? 'bg-blue-100 border-blue-500 text-blue-600'
                          : 'border-gray-200 hover:border-blue-300'
                      }`}
                    >
                      {slot}
                    </button>
                  ))}
              </div>
            </div>
          )}
        </div>

        {/* Booking Summary */}
        <div className="bg-white rounded-lg shadow p-6 h-fit">
          <h2 className="text-lg font-semibold mb-4">Booking Summary</h2>
          <div className="space-y-4">
            <div className="flex items-center space-x-3 text-gray-600">
              <Calendar className="h-5 w-5" />
              <span>{selectedDate || 'Select a date'}</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-600">
              <User className="h-5 w-5" />
              <span>
                {doctors.find((d) => d.id === selectedDoctor)?.name ||
                  'Select a doctor'}
              </span>
            </div>
            <div className="flex items-center space-x-3 text-gray-600">
              <Clock className="h-5 w-5" />
              <span>{selectedTime || 'Select a time'}</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-600">
              <MapPin className="h-5 w-5" />
              <span>Medical Center, Room 101</span>
            </div>

            <button
              className={`w-full py-2 px-4 rounded-lg ${
                selectedDate && selectedDoctor && selectedTime
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }`}
              disabled={!selectedDate || !selectedDoctor || !selectedTime}
            >
              Confirm Booking
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Appointments;