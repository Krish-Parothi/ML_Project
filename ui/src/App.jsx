import React, { useState, useEffect, useTransition } from 'react';
import './App.css';
import brainImg from './assets/brain.jpg';
import { 
  Plane, Sparkles, User, ShieldCheck, Activity, AlertTriangle, 
  CheckCircle2, Wifi, Star, Clock, Compass, ThumbsUp, ThumbsDown, RefreshCw
} from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

const INITIAL_PASSENGER = {
  gender: 'Female',
  customer_type: 'Returning',
  age: 42,
  type_of_travel: 'Business',
  travel_class: 'Business',
  flight_distance: 1850,
  departure_delay: 0,
  arrival_delay: 0,
  online_boarding: 5,
  inflight_wifi: 5,
  seat_comfort: 5,
  legroom_service: 5,
  cleanliness: 5,
  food_and_drink: 4,
  inflight_service: 5,
  inflight_entertainment: 5,
  baggage_handling: 5,
  onboard_service: 5,
  checkin_service: 5,
  gate_location: 4,
  ease_of_booking: 5,
  time_convenience: 4,
  model_name: 'LightGBM'
};

export default function App() {
  const [passenger, setPassenger] = useState(INITIAL_PASSENGER);
  const [selectedModel, setSelectedModel] = useState('LightGBM');
  const [prediction, setPrediction] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    runPrediction({ ...passenger, model_name: selectedModel });
  }, [passenger, selectedModel]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/`);
      if (res.ok) setBackendOnline(true);
      else setBackendOnline(false);
    } catch {
      setBackendOnline(false);
    }
  };

  const runPrediction = async (currentData) => {
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentData),
      });

      if (res.ok) {
        const data = await res.json();
        setBackendOnline(true);
        setPrediction(data);
      } else {
        throw new Error('Backend error');
      }
    } catch {
      setBackendOnline(false);
      simulateOfflinePrediction(currentData);
    }
  };

  // Offline fallback logic
  const simulateOfflinePrediction = (p) => {
    const isBiz = p.travel_class === 'Business';
    const delays = p.departure_delay + p.arrival_delay;
    const avgRating = (
      p.online_boarding * 2.5 +
      p.inflight_wifi * 2.5 +
      p.seat_comfort +
      p.cleanliness +
      p.inflight_entertainment +
      p.legroom_service
    ) / 9;

    let score = avgRating * 18 + (isBiz ? 15 : -10) - (delays > 45 ? 18 : 0);
    const isSat = score >= 50;

    let explanation = '';
    if (isSat) {
      if (delays > 60) {
        explanation = "Despite the flight delay, the passenger remains SATISFIED because premium service ratings (Online Boarding, Wifi, Seat Comfort) and Cabin Class heavily outweigh the delay.";
      } else {
        explanation = "The passenger is SATISFIED due to high service ratings and smooth overall flight experience.";
      }
    } else {
      if (delays > 30) {
        explanation = "The passenger is DISSATISFIED primarily due to flight delays coupled with sub-par in-flight experience.";
      } else {
        explanation = "The passenger is DISSATISFIED due to low ratings in key service touchpoints (Boarding, Wifi, or Amenities).";
      }
    }

    setPrediction({
      primary_model: selectedModel,
      prediction_label: isSat ? 'SATISFIED' : 'DISSATISFIED',
      is_satisfied: isSat,
      explanation: explanation,
      key_drivers: [
        { factor: 'Online Boarding', sentiment: p.online_boarding >= 4 ? 'positive' : 'negative', detail: `Rating: ${p.online_boarding}/5` },
        { factor: 'In-flight Wifi', sentiment: p.inflight_wifi >= 4 ? 'positive' : 'negative', detail: `Rating: ${p.inflight_wifi}/5` },
        { factor: 'Flight Delay', sentiment: delays > 40 ? 'negative' : 'positive', detail: `${delays}m total delay` },
        { factor: 'Travel Class', sentiment: isBiz ? 'positive' : 'negative', detail: p.travel_class }
      ],
      model_comparison: {
        LightGBM: { prediction: isSat ? 'SATISFIED' : 'DISSATISFIED', is_satisfied: isSat },
        XGBoost: { prediction: isSat ? 'SATISFIED' : 'DISSATISFIED', is_satisfied: isSat },
        'Random Forest': { prediction: isSat ? 'SATISFIED' : 'DISSATISFIED', is_satisfied: isSat }
      }
    });
  };

  const handleFieldChange = (field, value) => {
    startTransition(() => {
      setPassenger(prev => ({ ...prev, [field]: value }));
    });
  };

  const applyPreset = (type) => {
    if (type === 'satisfied_vip') {
      setPassenger({
        ...INITIAL_PASSENGER,
        gender: 'Female',
        travel_class: 'Business',
        type_of_travel: 'Business',
        online_boarding: 5,
        inflight_wifi: 5,
        seat_comfort: 5,
        legroom_service: 5,
        cleanliness: 5,
        departure_delay: 0,
        arrival_delay: 0
      });
    } else if (type === 'dissatisfied_economy') {
      setPassenger({
        ...INITIAL_PASSENGER,
        gender: 'Male',
        customer_type: 'First-time',
        age: 26,
        travel_class: 'Economy',
        type_of_travel: 'Personal',
        flight_distance: 650,
        departure_delay: 85,
        arrival_delay: 70,
        online_boarding: 1,
        inflight_wifi: 1,
        seat_comfort: 2,
        legroom_service: 1,
        cleanliness: 2,
        food_and_drink: 1,
        inflight_service: 2,
        inflight_entertainment: 1,
        baggage_handling: 2
      });
    } else if (type === 'borderline') {
      setPassenger({
        ...INITIAL_PASSENGER,
        gender: 'Male',
        customer_type: 'Returning',
        age: 35,
        travel_class: 'Economy Plus',
        type_of_travel: 'Business',
        flight_distance: 1200,
        departure_delay: 20,
        arrival_delay: 15,
        online_boarding: 3,
        inflight_wifi: 3,
        seat_comfort: 3,
        legroom_service: 3,
        cleanliness: 3,
        food_and_drink: 3,
        inflight_service: 3,
        inflight_entertainment: 3,
        baggage_handling: 3
      });
    }
  };

  return (
    <div className="app-container">
      {/* ─── Top Brand Header ─── */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brain-avatar-wrapper">
            <img src={brainImg} alt="Neural Brain" className="brain-avatar" />
          </div>
          <div>
            <div className="brand-title">
              NEURALFLIGHT <span style={{ fontSize: '13px', color: 'var(--neon-cyan)', fontWeight: 600 }}>AI</span>
            </div>
            <div className="brand-subtitle">Passenger Satisfaction Classifier (Feature-Based Verdicts)</div>
          </div>
        </div>

        <div className="header-status-badge">
          <span className="status-dot" style={{ backgroundColor: backendOnline ? '#10b981' : '#f59e0b' }} />
          {backendOnline ? 'FastAPI Backend Online (Port 8000)' : 'Local Classifier Active'}
        </div>
      </header>

      {/* ─── Control Bar ─── */}
      <div className="controls-bar">
        <div className="presets-container">
          <span className="presets-label">
            <Sparkles size={14} /> Quick Test Points:
          </span>
          <button className="preset-btn" onClick={() => applyPreset('satisfied_vip')}>
            🚀 VIP Business (Satisfied)
          </button>
          <button className="preset-btn" onClick={() => applyPreset('dissatisfied_economy')}>
            ⚠️ Frustrated Economy (Dissatisfied)
          </button>
          <button className="preset-btn" onClick={() => applyPreset('borderline')}>
            ⚖️ Economy Plus (Borderline)
          </button>
          <button className="preset-btn" onClick={() => setPassenger(INITIAL_PASSENGER)}>
            <RefreshCw size={12} /> Reset
          </button>
        </div>

        <div className="model-selector-container">
          <span style={{ fontSize: '11px', color: 'var(--text-dim)', fontWeight: 700, textTransform: 'uppercase' }}>Active Model:</span>
          {['LightGBM', 'XGBoost', 'Random Forest'].map(modelName => (
            <button
              key={modelName}
              className={`model-tab-btn ${selectedModel === modelName ? 'active' : ''}`}
              onClick={() => setSelectedModel(modelName)}
            >
              {modelName}
            </button>
          ))}
        </div>
      </div>

      {/* ─── Main Content Grid ─── */}
      <div className="main-grid">
        {/* Left: Input Form Panels */}
        <div className="form-column">
          {/* Card 1: Passenger Profile & Flight Info */}
          <div className="card">
            <div className="card-header">
              <User className="card-icon" size={18} />
              <div className="card-title">Passenger Demographics & Flight Details</div>
              <div className="card-subtitle">Flight Parameters</div>
            </div>

            <div className="fields-grid-3" style={{ marginBottom: '16px' }}>
              <div className="input-group">
                <label className="input-label">Gender</label>
                <select 
                  className="select-input"
                  value={passenger.gender}
                  onChange={e => handleFieldChange('gender', e.target.value)}
                >
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Loyalty Status</label>
                <select 
                  className="select-input"
                  value={passenger.customer_type}
                  onChange={e => handleFieldChange('customer_type', e.target.value)}
                >
                  <option value="Returning">Returning Member</option>
                  <option value="First-time">First-time Passenger</option>
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">
                  <span>Age</span>
                  <span className="input-value-badge">{passenger.age} yrs</span>
                </label>
                <input 
                  type="range" min="7" max="85"
                  className="slider-control"
                  value={passenger.age}
                  onChange={e => handleFieldChange('age', parseInt(e.target.value))}
                />
              </div>
            </div>

            <div className="fields-grid-2" style={{ marginBottom: '16px' }}>
              <div className="input-group">
                <label className="input-label">Travel Purpose</label>
                <select 
                  className="select-input"
                  value={passenger.type_of_travel}
                  onChange={e => handleFieldChange('type_of_travel', e.target.value)}
                >
                  <option value="Business">Business Travel</option>
                  <option value="Personal">Personal Travel</option>
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Cabin Class</label>
                <select 
                  className="select-input"
                  value={passenger.travel_class}
                  onChange={e => handleFieldChange('travel_class', e.target.value)}
                >
                  <option value="Business">Business Class</option>
                  <option value="Economy">Economy</option>
                  <option value="Economy Plus">Economy Plus</option>
                </select>
              </div>
            </div>

            <div className="fields-grid-3">
              <div className="input-group">
                <label className="input-label">
                  <span>Flight Distance</span>
                  <span className="input-value-badge">{passenger.flight_distance} mi</span>
                </label>
                <input 
                  type="range" min="100" max="4500" step="50"
                  className="slider-control"
                  value={passenger.flight_distance}
                  onChange={e => handleFieldChange('flight_distance', parseFloat(e.target.value))}
                />
              </div>

              <div className="input-group">
                <label className="input-label">
                  <span>Departure Delay</span>
                  <span className="input-value-badge" style={{ color: passenger.departure_delay > 30 ? 'var(--accent-red)' : 'var(--neon-cyan)' }}>
                    {passenger.departure_delay} min
                  </span>
                </label>
                <input 
                  type="range" min="0" max="300" step="5"
                  className="slider-control"
                  value={passenger.departure_delay}
                  onChange={e => handleFieldChange('departure_delay', parseFloat(e.target.value))}
                />
              </div>

              <div className="input-group">
                <label className="input-label">
                  <span>Arrival Delay</span>
                  <span className="input-value-badge" style={{ color: passenger.arrival_delay > 30 ? 'var(--accent-red)' : 'var(--neon-cyan)' }}>
                    {passenger.arrival_delay} min
                  </span>
                </label>
                <input 
                  type="range" min="0" max="300" step="5"
                  className="slider-control"
                  value={passenger.arrival_delay}
                  onChange={e => handleFieldChange('arrival_delay', parseFloat(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* Card 2: Digital & Airport Touchpoints */}
          <div className="card">
            <div className="card-header">
              <Wifi className="card-icon" size={18} />
              <div className="card-title">Digital & Airport Experience (Top Predictors)</div>
              <div className="card-subtitle">Ranked #1 & #2 in importance</div>
            </div>

            <div className="ratings-grid">
              <RatingSelector 
                label="Online Boarding (#1 Driver)" 
                value={passenger.online_boarding} 
                onChange={v => handleFieldChange('online_boarding', v)} 
              />
              <RatingSelector 
                label="In-flight Wifi (#2 Driver)" 
                value={passenger.inflight_wifi} 
                onChange={v => handleFieldChange('inflight_wifi', v)} 
              />
              <RatingSelector 
                label="Check-in Service" 
                value={passenger.checkin_service} 
                onChange={v => handleFieldChange('checkin_service', v)} 
              />
              <RatingSelector 
                label="Ease of Online Booking" 
                value={passenger.ease_of_booking} 
                onChange={v => handleFieldChange('ease_of_booking', v)} 
              />
              <RatingSelector 
                label="Gate Location Convenience" 
                value={passenger.gate_location} 
                onChange={v => handleFieldChange('gate_location', v)} 
              />
              <RatingSelector 
                label="Flight Time Convenience" 
                value={passenger.time_convenience} 
                onChange={v => handleFieldChange('time_convenience', v)} 
              />
            </div>
          </div>

          {/* Card 3: In-Flight Amenities & Comfort */}
          <div className="card">
            <div className="card-header">
              <Plane className="card-icon" size={18} />
              <div className="card-title">In-Cabin Comfort & Amenities</div>
              <div className="card-subtitle">1 to 5 Stars</div>
            </div>

            <div className="ratings-grid">
              <RatingSelector 
                label="Seat Comfort" 
                value={passenger.seat_comfort} 
                onChange={v => handleFieldChange('seat_comfort', v)} 
              />
              <RatingSelector 
                label="Leg Room Service" 
                value={passenger.legroom_service} 
                onChange={v => handleFieldChange('legroom_service', v)} 
              />
              <RatingSelector 
                label="Cabin Cleanliness" 
                value={passenger.cleanliness} 
                onChange={v => handleFieldChange('cleanliness', v)} 
              />
              <RatingSelector 
                label="Food & Drink" 
                value={passenger.food_and_drink} 
                onChange={v => handleFieldChange('food_and_drink', v)} 
              />
              <RatingSelector 
                label="In-flight Service" 
                value={passenger.inflight_service} 
                onChange={v => handleFieldChange('inflight_service', v)} 
              />
              <RatingSelector 
                label="In-flight Entertainment" 
                value={passenger.inflight_entertainment} 
                onChange={v => handleFieldChange('inflight_entertainment', v)} 
              />
              <RatingSelector 
                label="Baggage Handling" 
                value={passenger.baggage_handling} 
                onChange={v => handleFieldChange('baggage_handling', v)} 
              />
              <RatingSelector 
                label="On-board Service" 
                value={passenger.onboard_service} 
                onChange={v => handleFieldChange('onboard_service', v)} 
              />
            </div>
          </div>
        </div>

        {/* Right: Clean Prediction Output */}
        <div className="prediction-column">
          {prediction && (
            <>
              {/* Clean Verdict Card */}
              <div className={`hero-verdict-card ${prediction.is_satisfied ? 'verdict-satisfied' : 'verdict-dissatisfied'}`}>
                <div className="verdict-badge-header">
                  {prediction.is_satisfied ? <ThumbsUp size={16} /> : <ThumbsDown size={16} />}
                  <span>Direct Model Verdict</span>
                </div>

                <div className="verdict-status-title" style={{ fontSize: '32px', margin: '20px 0' }}>
                  {prediction.is_satisfied ? 'PASSENGER IS SATISFIED' : 'PASSENGER IS DISSATISFIED'}
                </div>

                {/* Plain-English Feature Reason */}
                <div style={{
                  background: 'rgba(0, 0, 0, 0.35)',
                  padding: '14px',
                  borderRadius: '12px',
                  fontSize: '13px',
                  lineHeight: '1.5',
                  color: '#e2e8f0',
                  border: '1px solid rgba(255, 255, 255, 0.08)'
                }}>
                  {prediction.explanation}
                </div>
              </div>

              {/* Cross-Model Consensus (Pure Verdicts) */}
              <div className="consensus-card">
                <div className="consensus-header">
                  <span>Model Consensus Comparison</span>
                  <span style={{ fontSize: '11px', color: 'var(--neon-cyan)', fontWeight: 600 }}>Pure Verdicts</span>
                </div>

                <div className="consensus-models-list">
                  {Object.entries(prediction.model_comparison || {}).map(([mName, res]) => (
                    <div key={mName} className="consensus-item">
                      <div className="consensus-model-info">
                        <span className="consensus-model-name">{mName}</span>
                      </div>
                      <div className="consensus-vote">
                        <span className={`vote-badge ${res.is_satisfied ? 'vote-satisfied' : 'vote-dissatisfied'}`} style={{ fontSize: '13px', padding: '6px 14px' }}>
                          {res.prediction}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Key Contributing Feature Badges */}
              <div className="drivers-card">
                <div style={{ fontSize: '13px', fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Compass size={15} color="var(--neon-cyan)" /> Key Features Driving This Verdict
                </div>

                <div className="drivers-list">
                  {prediction.key_drivers?.map((driver, idx) => (
                    <div 
                      key={idx} 
                      className={`driver-pill ${driver.sentiment === 'positive' ? 'driver-positive' : 'driver-negative'}`}
                    >
                      <span style={{ fontWeight: 600 }}>{driver.factor}</span>
                      <span style={{ fontSize: '12px', opacity: 0.9 }}>{driver.detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Sub-component: 1-5 Star Selector ───
function RatingSelector({ label, value, onChange }) {
  return (
    <div className="rating-item">
      <div className="rating-item-header">
        <span>{label}</span>
        <span style={{ color: 'var(--neon-cyan)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
          {value} / 5
        </span>
      </div>
      <div className="star-rating-selector">
        {[1, 2, 3, 4, 5].map(rating => (
          <button
            key={rating}
            type="button"
            className={`star-btn ${value === rating ? 'active' : ''}`}
            onClick={() => onChange(rating)}
          >
            {rating} ★
          </button>
        ))}
      </div>
    </div>
  );
}
