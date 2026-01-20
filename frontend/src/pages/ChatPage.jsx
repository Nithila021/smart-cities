import React, { useState, useRef, useEffect } from 'react';
import { Send, AlertTriangle } from 'lucide-react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';
import SafetyMap from '../components/SafetyMap';

	// Function to safely escape any raw HTML before formatting
	const escapeHtml = (text) => {
	  return text
	    .replace(/&/g, '&amp;')
	    .replace(/</g, '&lt;')
	    .replace(/>/g, '&gt;')
	    .replace(/"/g, '&quot;')
	    .replace(/'/g, '&#39;');
	};

	// Function to convert markdown-like syntax to HTML (bold/italic/paragraphs)
	// NOTE: We always escape first so that only the tags we intentionally add
	// (`<strong>`, `<em>`, `<p>`) are rendered as HTML. This reduces XSS risk
	// even though we use `dangerouslySetInnerHTML` when rendering.
	const formatMessage = (text) => {
	  if (!text) return '';

	  const safeText = escapeHtml(text);
	  
	  let formattedText = safeText.replace(/\*\*\*(.*?)\*\*\*/g, '<strong>$1</strong>');
	  formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
	  formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
	  formattedText = formattedText.split('\n\n').map(para => 
	    `<p>${para}</p>`
	  ).join('');
	  
	  return formattedText;
	};

const CrimeChart = ({ crimeData }) => {
  // Sort and get top 10 crimes only
  const sortedCrimes = Object.entries(crimeData || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);
  
  const data = {
    labels: sortedCrimes.map(([crime]) => crime),
    datasets: [{
      label: 'Crime Distribution',
      data: sortedCrimes.map(([, count]) => count),
      backgroundColor: '#ec4899',
      borderColor: '#db2777',
      borderWidth: 1
    }]
  };

  return (
    <div className="chart-container" style={{ height: '150px', width: '100%', overflow: 'hidden' }}>
      <Bar 
        data={data}
        options={{
          maintainAspectRatio: false,
          responsive: true,
          plugins: { 
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => {
                  // Wrap long crime names in tooltip
                  return items[0].label.length > 30 
                    ? items[0].label.substring(0, 30) + '...' 
                    : items[0].label;
                }
              }
            }
          },
          scales: {
            y: { 
              beginAtZero: true, 
              grid: { color: '#f3f4f6' },
              ticks: { font: { size: 11 } }
            },
            x: { 
              grid: { display: false },
              ticks: {
                font: { size: 10 },
                maxRotation: 45,
                minRotation: 45,
                autoSkip: false,
                callback: function(value) {
                  // Truncate long labels
                  const label = this.getLabelForValue(value);
                  return label.length > 15 ? label.substring(0, 15) + '...' : label;
                }
              }
            }
          }
        }}
      />
    </div>
  );
};

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [demographicsInput, setDemographicsInput] = useState('');
  const [showDemographicsInput, setShowDemographicsInput] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [visualizationData, setVisualizationData] = useState({
    clusters: null,
    crimeStats: null,
    center: null
  });
  const [showEmergencyModal, setShowEmergencyModal] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [expandedView, setExpandedView] = useState(null); // 'map' or 'chart'
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    const userMessage = inputMessage.trim();
    const demographics = demographicsInput.trim();
    
    // Add user message and demographics if provided
    const newMessages = [{ text: userMessage, sender: 'user' }];
    if (demographics) {
      newMessages.push({ text: `👥 Group: ${demographics}`, sender: 'user', isSubtext: true });
    }
    setMessages([...messages, ...newMessages]);
    setInputMessage('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: userMessage,
          demographics: demographics 
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Backend error:', errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('Backend response:', data);

      // Fetch cluster data
      const [clusters, crimeStats] = await Promise.all([
        fetch('http://localhost:5000/api/dbscan_clusters').then(res => res.json()),
        fetch('http://localhost:5000/api/demographic_zones').then(res => res.json())
      ]);

      setVisualizationData({
        clusters: {
          dbscan: clusters.clusters,
          demographic: crimeStats.zones,
          density: []
        },
        crimeStats: data.crime_types,
        center: [data.lat, data.lon]
      });

      // Add response to messages
      setMessages(prev => [...prev, { 
        text: data.text, 
        sender: 'bot',
        graph: data.graph 
      }]);
      // Clear demographics after successful send
      setDemographicsInput('');
    } catch (error) {
      console.error('Full error details:', error);
      setMessages(prev => [...prev, { 
        text: 'Sorry, there was an error processing your request.', 
        sender: 'bot' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleEmergencyClick = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude
          });
          setShowEmergencyModal(true);
        },
        (error) => {
          alert('Unable to get your location. Please enable location services.');
          console.error('Geolocation error:', error);
        }
      );
    } else {
      alert('Geolocation is not supported by your browser.');
    }
  };

  const handleSendEmergencyAlert = () => {
    if (userLocation) {
      const message = `🚨 EMERGENCY ALERT 🚨\nI need help!\nMy location: https://www.google.com/maps?q=${userLocation.lat},${userLocation.lon}\nLat: ${userLocation.lat.toFixed(6)}, Lon: ${userLocation.lon.toFixed(6)}`;
      
      // Copy to clipboard
      navigator.clipboard.writeText(message).then(() => {
        alert('Emergency location copied to clipboard! Share it with your contacts via SMS, WhatsApp, or any messaging app.');
        setShowEmergencyModal(false);
      }).catch((err) => {
        console.error('Failed to copy:', err);
        alert('Failed to copy. Please manually share your location.');
      });
    }
  };
  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1 className="header-title">Safety Analysis Chat</h1>
      </div>
      
      <div className="messages-area">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p>Enter an amenity and address, separated by a comma.</p>
              <p className="example">Example: "Restaurant, 123 Main St, New York"</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div 
                key={index} 
                className={`message-row ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
              >
                <div className={`message-bubble ${message.sender === 'user' ? 'user-bubble' : 'bot-bubble'}`}>
                  {message.sender === 'bot' ? (
                    <div 
                      className="message-text formatted"
                      dangerouslySetInnerHTML={{ __html: formatMessage(message.text) }}
                    />
                  ) : (
                    <p className="message-text">{message.text}</p>
                  )}
                  {/* Removed inline graph - use expand button instead */}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message-row bot-message">
              <div className="loading-bubble">
                <div className="loading-spinner"></div>
                <span className="loading-text">Analyzing...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {visualizationData.center && (
        <div className="visualization-section">
          <div className="map-frame">
            <button
              onClick={() => setExpandedView('map')}
              style={{
                position: 'absolute',
                top: '8px',
                right: '8px',
                zIndex: 1000,
                background: 'white',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                padding: '6px 10px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: '600'
              }}
            >
              🔍 Expand Map
            </button>
            <SafetyMap 
              clusters={visualizationData.clusters} 
              center={visualizationData.center}
            />
          </div>
          {visualizationData.crimeStats && (
            <div className="chart-frame">
              <button
                onClick={() => setExpandedView('chart')}
                style={{
                  position: 'absolute',
                  top: '8px',
                  right: '8px',
                  zIndex: 1000,
                  background: 'white',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: '600'
                }}
              >
                📊 Expand Chart
              </button>
              <CrimeChart crimeData={visualizationData.crimeStats} />
            </div>
          )}
        </div>
      )}

      {/* Expanded View Modal */}
      {expandedView && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.8)',
            zIndex: 2000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px'
          }}
          onClick={() => setExpandedView(null)}
        >
          <div 
            style={{
              background: 'white',
              borderRadius: '12px',
              width: '90%',
              height: '90%',
              position: 'relative',
              overflow: 'hidden'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setExpandedView(null)}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                zIndex: 3000,
                background: '#dc2626',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '40px',
                height: '40px',
                fontSize: '24px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              ×
            </button>
            <div style={{ width: '100%', height: '100%', padding: '20px' }}>
              {expandedView === 'map' && (
                <SafetyMap 
                  clusters={visualizationData.clusters} 
                  center={visualizationData.center}
                />
              )}
              {expandedView === 'chart' && visualizationData.crimeStats && (
                <div style={{ height: '100%' }}>
                  <h2 style={{ marginBottom: '20px', color: '#db2777' }}>Crime Distribution Analysis</h2>
                  <CrimeChart crimeData={visualizationData.crimeStats} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="input-container">
        {/* Demographics Toggle */}
        <div style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setShowDemographicsInput(!showDemographicsInput)}
            style={{
              padding: '6px 12px',
              fontSize: '12px',
              background: showDemographicsInput ? '#ec4899' : '#f3f4f6',
              color: showDemographicsInput ? 'white' : '#374151',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            {showDemographicsInput ? '✓ Custom Group' : '+ Add Group Details'}
          </button>
          {showDemographicsInput && (
            <span style={{ fontSize: '11px', color: '#6b7280' }}>
              e.g., "3 women + 2 children" or "1 queer couple"
            </span>
          )}
        </div>

        {/* Demographics Input */}
        {showDemographicsInput && (
          <div style={{ marginBottom: '8px' }}>
            <input
              type="text"
              value={demographicsInput}
              onChange={(e) => setDemographicsInput(e.target.value)}
              placeholder="Describe your group (e.g., 2 white women, 1 black man, 3 children)"
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid #d1d5db',
                borderRadius: '8px',
                fontSize: '14px'
              }}
            />
          </div>
        )}

        <div className="input-wrapper">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Type amenity, address (e.g., Restaurant, 123 Main St, New York)"
            className="message-input"
            rows={1}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className={`send-button ${!inputMessage.trim() || isLoading ? 'disabled' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m22 2-7 20-4-9-9-4Z"/>
              <path d="M22 2 11 13"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Emergency Button */}
      <button 
        className="emergency-button" 
        onClick={handleEmergencyClick}
        title="Emergency - Share Location"
      >
        <AlertTriangle size={32} />
      </button>

      {/* Emergency Modal */}
      {showEmergencyModal && (
        <div className="emergency-modal" onClick={() => setShowEmergencyModal(false)}>
          <div className="emergency-modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>🚨 Emergency Alert</h2>
            <p>Your current location will be copied to your clipboard.</p>
            <p>You can then share it with friends, family, or emergency services via:</p>
            <ul style={{ marginLeft: '20px', marginBottom: '12px' }}>
              <li>SMS / Text Message</li>
              <li>WhatsApp</li>
              <li>Any messaging app</li>
            </ul>
            {userLocation && (
              <div className="location-info">
                <strong>Your Location:</strong><br />
                Lat: {userLocation.lat.toFixed(6)}<br />
                Lon: {userLocation.lon.toFixed(6)}<br />
                <a 
                  href={`https://www.google.com/maps?q=${userLocation.lat},${userLocation.lon}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: '#dc2626', textDecoration: 'underline' }}
                >
                  View on Google Maps
                </a>
              </div>
            )}
            <div className="emergency-modal-buttons">
              <button className="send-button" onClick={handleSendEmergencyAlert}>
                Copy & Share Location
              </button>
              <button className="cancel-button" onClick={() => setShowEmergencyModal(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}