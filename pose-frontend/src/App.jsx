import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [userName, setUserName] = useState('');
  const [posture, setPosture] = useState('Unknown');
  const [postureScore, setPostureScore] = useState(0);
  const [confidenceHistory, setConfidenceHistory] = useState([]);
  const [timestamp, setTimestamp] = useState('');
  const [historyTimestamps, setHistoryTimestamps] = useState([]);
  const [suggestion, setSuggestion] = useState('');
  const [showAbout, setShowAbout] = useState(false);
  const [hasName, setHasName] = useState(false);
  const [currentTab, setCurrentTab] = useState('home');
  const [isDetecting, setIsDetecting] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [demoMode, setDemoMode] = useState(true);
  const [detectionInterval, setDetectionInterval] = useState(null);
  const [isAutoDetecting, setIsAutoDetecting] = useState(false);
  const THRESHOLD = 80;

  // Demo data for testing UI
  useEffect(() => {
    if (hasName && demoMode) {
      // Add some initial demo data
      const demoData = [65, 72, 58, 85, 91, 77, 82, 69, 95, 88, 73, 79, 84, 92, 67];
      const demoTimes = demoData.map((_, i) => {
        const time = new Date();
        time.setMinutes(time.getMinutes() - (demoData.length - i));
        return time.toLocaleTimeString();
      });
      setConfidenceHistory(demoData);
      setHistoryTimestamps(demoTimes);
      setPostureScore(demoData[demoData.length - 1]);
      setPosture('Sitting Upright');
      setSuggestion('Great posture! Keep your shoulders relaxed and maintain this position.');
    }
  }, [hasName, demoMode]);


  useEffect(() => {
    // Start camera after entering name and main UI is visible
    if (!hasName) return;
    import('./camera').then(mod => {
      mod.startCamera('videoElement');
    });
  }, [hasName]);

  useEffect(() => {
    if (!hasName) return;
    const interval = setInterval(() => {
      const now = new Date();
      setTimestamp(now.toLocaleTimeString());
    }, 1000);
    return () => clearInterval(interval);
  }, [hasName]);

  // Cleanup auto-detection interval on unmount
  useEffect(() => {
    return () => {
      if (detectionInterval) {
        clearInterval(detectionInterval);
      }
    };
  }, [detectionInterval]);

  // Fetch posture and suggestion from backend
  async function fetchPostureData() {
    setIsDetecting(true);
    
    if (demoMode) {
      // Simulate API call with random data
      setTimeout(() => {
        const randomScore = Math.floor(Math.random() * 100);
        const postures = ['Sitting Upright', 'Slouching', 'Leaning Forward', 'Good Posture', 'Head Forward', 'Shoulders Rounded'];
        const suggestions = [
          'Excellent posture! Keep it up!',
          'Try to straighten your back and pull shoulders back.',
          'Adjust your monitor height to reduce neck strain.',
          'Perfect alignment detected. Well done!',
          'Pull your head back and align with your spine.',
          'Roll your shoulders back and down for better posture.'
        ];
        
        const newPosture = postures[Math.floor(Math.random() * postures.length)];
        const newSuggestion = suggestions[Math.floor(Math.random() * suggestions.length)];
        
        setPosture(newPosture);
        setPostureScore(randomScore);
        setConfidenceHistory(prev => [...prev.slice(-29), randomScore]);
        setHistoryTimestamps(prev => [...prev.slice(-29), new Date().toLocaleTimeString()]);
        setSuggestion(newSuggestion);
        
        // Add notification for good/bad posture
        if (randomScore >= THRESHOLD) {
          addNotification('Good posture detected! 👍', 'success');
        } else if (randomScore < 50) {
          addNotification('Please adjust your posture', 'warning');
        } else {
          addNotification('Posture analysis complete', 'success');
        }
        
        setIsDetecting(false);
      }, 2000);
      return;
    }

    // Replace with actual backend endpoint
    try {
      const res = await fetch('/api/posture');
      const data = await res.json();
      setPosture(data.posture);
      setPostureScore(data.confidence);
      setConfidenceHistory(prev => [...prev.slice(-29), data.confidence]);
      setHistoryTimestamps(prev => [...prev.slice(-29), new Date().toLocaleTimeString()]);
      setSuggestion(data.suggestion);
      
      // Add notification for good/bad posture
      if (data.confidence >= THRESHOLD) {
        addNotification('Good posture detected! 👍', 'success');
      } else if (data.confidence < 50) {
        addNotification('Please adjust your posture', 'warning');
      }
    } catch (err) {
      setPosture('Unknown');
      setPostureScore(0);
      setSuggestion('Unable to fetch suggestion.');
      addNotification('Connection error. Please try again.', 'error');
    }
    setTimeout(() => setIsDetecting(false), 1000);
  }

  function addNotification(message, type) {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 4000);
  }

  function generateMoreDemoData() {
    const newData = Array.from({ length: 10 }, () => Math.floor(Math.random() * 100));
    const newTimes = newData.map((_, i) => {
      const time = new Date();
      time.setSeconds(time.getSeconds() + i * 5);
      return time.toLocaleTimeString();
    });
    
    setConfidenceHistory(prev => [...prev, ...newData].slice(-30));
    setHistoryTimestamps(prev => [...prev, ...newTimes].slice(-30));
    addNotification('Generated 10 new data points', 'success');
  }

  function clearData() {
    setConfidenceHistory([]);
    setHistoryTimestamps([]);
    setPostureScore(0);
    setPosture('Unknown');
    setSuggestion('');
    addNotification('All data cleared', 'warning');
  }

  function startAutoDetection() {
    if (isAutoDetecting) return;
    
    setIsAutoDetecting(true);
    addNotification('Auto-detection started', 'success');
    
    const interval = setInterval(() => {
      fetchPostureData();
    }, 5000); // Run every 5 seconds
    
    setDetectionInterval(interval);
  }

  function stopAutoDetection() {
    if (detectionInterval) {
      clearInterval(detectionInterval);
      setDetectionInterval(null);
    }
    setIsAutoDetecting(false);
    setIsDetecting(false);
    addNotification('Auto-detection stopped', 'warning');
  }

  function testNotification() {
    const notifications = [
      { message: 'Test notification! 🎉', type: 'success' },
      { message: 'Warning notification test ⚠️', type: 'warning' },
      { message: 'Error notification test ❌', type: 'error' }
    ];
    const randomNotif = notifications[Math.floor(Math.random() * notifications.length)];
    addNotification(randomNotif.message, randomNotif.type);
  }
  
  if (!hasName) {
    return (
      <div className="welcome">
        <div className="welcome-card">
          <div className="app-badge" aria-hidden>
            <span>🧍‍♂️</span>
          </div>
          <h1 className="gradient-text">Welcome to Pose Detection</h1>
          <p className="subtitle">Enter your name to continue</p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (userName.trim()) setHasName(true);
            }}
            className="name-form"
          >
            <input
              type="text"
              placeholder="Your name"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              className="text-input"
              autoFocus
            />
            <button type="submit" className="btn btn-primary" disabled={!userName.trim()}>
              Continue
            </button>
          </form>
          <div className="tip">Press Enter to continue</div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <header className="header">
        <div className="user-info">
          <div className="user-name">{userName}</div>
          {isAutoDetecting && (
            <div className="detection-status">
              <span className="status-indicator pulse"></span>
              Auto-detecting every 5s
            </div>
          )}
        </div>
        <div className="header-tabs">
          <button 
            className={`tab-btn ${currentTab === 'home' ? 'active' : ''}`}
            onClick={() => setCurrentTab('home')}
          >
            Home
          </button>
          <button 
            className={`tab-btn ${currentTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setCurrentTab('analytics')}
          >
            Analytics
          </button>
        </div>
        <div className="top-actions">
          <button 
            className={`btn btn-primary ${isDetecting ? 'loading' : ''}`} 
            onClick={isAutoDetecting ? stopAutoDetection : startAutoDetection}
            disabled={isDetecting && !isAutoDetecting}
          >
            {isDetecting && !isAutoDetecting ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : isAutoDetecting ? (
              '⏹️ Stop Detection'
            ) : (
              '▶️ Start Auto Detection'
            )}
          </button>
          <button 
            className="btn btn-ghost" 
            onClick={fetchPostureData}
            disabled={isDetecting}
          >
            {isDetecting ? 'Analyzing...' : '🔍 Single Scan'}
          </button>
          <button 
            className="btn btn-ghost" 
            onClick={() => {
              setDemoMode(!demoMode);
              addNotification(demoMode ? 'Demo mode disabled' : 'Demo mode enabled', 'success');
            }}
          >
            {demoMode ? '🎮 Demo Mode' : '🔗 Live Mode'}
          </button>
          <button className="btn btn-ghost" onClick={() => setShowAbout(true)}>About This App</button>
        </div>
      </header>
      
      {currentTab === 'home' && (
        <main className="main-content">
          <section className="camera-section">
            <div className="camera-feed">
              {/* Camera feed will be shown here */}
              <video autoPlay playsInline width="480" height="360" id="videoElement"></video>
              {/* Skeletal overlay will be drawn here later */}
              <canvas id="overlay" width="480" height="360" className="overlay"></canvas>
              {/* Timestamp overlay */}
              <div className="timestamp-box">🕒 {timestamp}</div>
            </div>
          </section>
          <aside className="info-section">
            <div className="threshold-box">
              <h2>Posture Score Threshold</h2>
              <div className="threshold-value">{THRESHOLD}%</div>
              <div className="threshold-desc">Anything above this threshold is considered a good posture!</div>
            </div>
            <div className="posture-info">
              <h2>Current Posture</h2>
              <div className="posture-row">
                <span className={`status-pill ${posture !== 'Unknown' ? 'ok' : 'idle'} ${isDetecting ? 'pulse' : ''}`}>
                  {isDetecting ? 'Analyzing' : posture !== 'Unknown' ? 'Detecting' : 'Idle'}
                </span>
                <span className="posture-name">{posture}</span>
              </div>
              <div className="confidence">
                <div className="label">
                  Posture Score
                  <span className={`value ${postureScore >= THRESHOLD ? 'good' : postureScore < 50 ? 'poor' : ''}`}>
                    {postureScore}%
                  </span>
                </div>
                <div className="bar">
                  <div 
                    className={`fill ${postureScore >= THRESHOLD ? 'excellent' : postureScore < 50 ? 'poor' : 'fair'}`} 
                    style={{ width: `${Math.min(Math.max(postureScore, 0), 100)}%` }} 
                  />
                </div>
              </div>
              <div className="confidence-trend">
                {confidenceHistory.length > 1 && (
                  <span className={`trend ${confidenceHistory[confidenceHistory.length - 1] > confidenceHistory[confidenceHistory.length - 2] ? 'up' : 'down'}`}>
                    {confidenceHistory[confidenceHistory.length - 1] > confidenceHistory[confidenceHistory.length - 2] ? '📈 Improving' : '📉 Declining'}
                  </span>
                )}
              </div>
            </div>
            <div className="suggestion-info">
              <h2>Suggestion</h2>
              <p>{suggestion || 'No suggestion available.'}</p>
            </div>
            <div className="analytics-preview">
              <button 
                className="btn btn-ghost analytics-btn"
                onClick={() => setCurrentTab('analytics')}
              >
                📊 View Full Analytics
              </button>
            </div>
            <div className="demo-controls">
              <h3>Demo Controls</h3>
              <div className="control-buttons">
                <button className="btn btn-ghost demo-btn" onClick={generateMoreDemoData}>
                  📈 Generate Data
                </button>
                <button className="btn btn-ghost demo-btn" onClick={clearData}>
                  🗑️ Clear All
                </button>
                <button 
                  className="btn btn-ghost demo-btn" 
                  onClick={testNotification}
                >
                  🔔 Test Notification
                </button>
              </div>
            </div>
          </aside>
        </main>
      )}

      {currentTab === 'analytics' && (
        <main className="analytics-content">
          <div className="analytics-header">
            <h1>Analytics & History</h1>
            <p>Detailed view of your posture scores over time</p>
          </div>
          <div className="analytics-grid">
            <div className="large-graph-section">
              <LargePostureScoreGraph 
                data={confidenceHistory} 
                timestamps={historyTimestamps} 
                threshold={THRESHOLD} 
              />
            </div>
            <div className="data-table-section">
              <h3>Session Data</h3>
              <DataTable 
                data={confidenceHistory} 
                timestamps={historyTimestamps} 
                threshold={THRESHOLD}
              />
            </div>
          </div>
        </main>
      )}
  {showAbout && (
        <div className="modal">
          <div className="modal-content">
            <h2>About This App</h2>
            <p>This app uses pose detection to help you maintain perfect posture. The backend AI will analyze your joint coordinates and provide feedback and suggestions for improvement.</p>
            <button onClick={() => setShowAbout(false)}>Close</button>
          </div>
        </div>
  )}
      <footer className="footer">Pose Detection UI</footer>
    </div>
  );
}

function PostureScoreGraph({ data, timestamps, threshold }) {
  // SVG graph, 30 points max, with axis and timestamp labels
  const width = 320, height = 120, pad = 32;
  const points = data.length ? data : Array(30).fill(0);
  const timeLabels = timestamps.length ? timestamps : Array(points.length).fill('');
  const step = (width - pad * 2) / (points.length - 1);
  const maxY = 100;
  const graphPoints = points.map((v, i) => `${pad + i * step},${height - pad - (v / maxY) * (height - pad * 2)}`).join(' ');
  const thresholdY = height - pad - (threshold / maxY) * (height - pad * 2);
  return (
    <div className="graph-container">
      <svg width={width} height={height} className="graph-svg">
        {/* Axis */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9da3af" strokeWidth="2" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9da3af" strokeWidth="2" />
        {/* Threshold line */}
        <line x1={pad} y1={thresholdY} x2={width - pad} y2={thresholdY} stroke="#22d3ee" strokeDasharray="4" strokeWidth="2" />
        {/* Graph line */}
        <polyline points={graphPoints} fill="none" stroke="#7c3aed" strokeWidth="3" />
        {/* Points */}
        {points.map((v, i) => (
          <circle key={i} cx={pad + i * step} cy={height - pad - (v / maxY) * (height - pad * 2)} r="4" fill="#22c55e" />
        ))}
        {/* Y axis labels */}
        {[0, 20, 40, 60, 80, 100].map((y) => (
          <text key={y} x={pad - 8} y={height - pad - (y / maxY) * (height - pad * 2) + 5} fontSize="10" fill="#9da3af" textAnchor="end">{y}</text>
        ))}
        {/* X axis timestamp labels (show every 5th) */}
        {timeLabels.map((t, i) => (
          i % 5 === 0 && t ? (
            <text key={i} x={pad + i * step} y={height - pad + 18} fontSize="10" fill="#9da3af" textAnchor="middle">{t}</text>
          ) : null
        ))}
      </svg>
      <div className="graph-legend">
        <span className="legend-line" style={{ background: '#7c3aed' }} /> Posture Score
        <span className="legend-dot" style={{ background: '#22c55e' }} /> Data Point
        <span className="legend-threshold" style={{ background: '#22d3ee' }} /> Threshold
      </div>
    </div>
  );
}

function LargePostureScoreGraph({ data, timestamps, threshold }) {
  const width = 800, height = 300, pad = 60;
  const points = data.length ? data : Array(30).fill(0);
  const timeLabels = timestamps.length ? timestamps : Array(points.length).fill('');
  const step = (width - pad * 2) / Math.max(points.length - 1, 1);
  const maxY = 100;
  const graphPoints = points.map((v, i) => `${pad + i * step},${height - pad - (v / maxY) * (height - pad * 2)}`).join(' ');
  const thresholdY = height - pad - (threshold / maxY) * (height - pad * 2);
  
  return (
    <div className="large-graph-container">
      <h3>Posture Score Timeline</h3>
      <svg width={width} height={height} className="large-graph-svg">
        {/* Grid lines */}
        {[0, 20, 40, 60, 80, 100].map((y) => (
          <line key={y} x1={pad} y1={height - pad - (y / maxY) * (height - pad * 2)} 
                x2={width - pad} y2={height - pad - (y / maxY) * (height - pad * 2)} 
                stroke="rgba(148,163,184,0.1)" strokeWidth="1" />
        ))}
        {/* Axis */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#94a3b8" strokeWidth="3" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#94a3b8" strokeWidth="3" />
        {/* Threshold line */}
        <line x1={pad} y1={thresholdY} x2={width - pad} y2={thresholdY} stroke="#06b6d4" strokeDasharray="8" strokeWidth="3" />
        <text x={width - pad + 10} y={thresholdY + 5} fontSize="12" fill="#06b6d4" fontWeight="600">Threshold: {threshold}%</text>
        {/* Graph area fill */}
        <polygon points={`${pad},${height - pad} ${graphPoints} ${pad + (points.length - 1) * step},${height - pad}`} 
                 fill="url(#gradient)" opacity="0.3" />
        {/* Graph line */}
        <polyline points={graphPoints} fill="none" stroke="#6366f1" strokeWidth="4" />
        {/* Points */}
        {points.map((v, i) => (
          <circle key={i} cx={pad + i * step} cy={height - pad - (v / maxY) * (height - pad * 2)} 
                  r="6" fill="#10b981" stroke="#ffffff" strokeWidth="2" />
        ))}
        {/* Y axis labels */}
        {[0, 20, 40, 60, 80, 100].map((y) => (
          <text key={y} x={pad - 12} y={height - pad - (y / maxY) * (height - pad * 2) + 5} 
                fontSize="14" fill="#94a3b8" textAnchor="end" fontWeight="500">{y}%</text>
        ))}
        {/* X axis timestamp labels (show every 3rd to avoid cramping) */}
        {timeLabels.map((t, i) => (
          i % 3 === 0 && t ? (
            <text key={i} x={pad + i * step} y={height - pad + 25} fontSize="12" fill="#94a3b8" 
                  textAnchor="middle" fontWeight="500">{t}</text>
          ) : null
        ))}
        {/* Gradient definition */}
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.8"/>
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.2"/>
          </linearGradient>
        </defs>
      </svg>
      <div className="large-graph-legend">
        <div className="legend-item">
          <span className="legend-line" style={{ background: '#6366f1' }} /> Posture Score Level
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#10b981' }} /> Data Points
        </div>
        <div className="legend-item">
          <span className="legend-threshold" style={{ background: '#06b6d4' }} /> Good Posture Threshold
        </div>
      </div>
    </div>
  );
}

function DataTable({ data, timestamps, threshold }) {
  const combinedData = data.map((postureScore, index) => ({
    id: index + 1,
    timestamp: timestamps[index] || 'N/A',
    postureScore: postureScore,
    status: postureScore >= threshold ? 'Good' : 'Needs Improvement'
  })).reverse(); // Show most recent first

  return (
    <div className="data-table-container">
      <div className="table-header">
        <div className="table-stats">
          <div className="stat">
            <span className="stat-label">Total Records:</span>
            <span className="stat-value">{data.length}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Average Score:</span>
            <span className="stat-value">{data.length ? Math.round(data.reduce((a, b) => a + b, 0) / data.length) : 0}%</span>
          </div>
          <div className="stat">
            <span className="stat-label">Good Posture Sessions:</span>
            <span className="stat-value">{data.filter(v => v >= threshold).length}</span>
          </div>
        </div>
      </div>
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>Posture Score</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {combinedData.slice(0, 20).map((row) => (
              <tr key={row.id} className={row.postureScore >= threshold ? 'good-posture' : 'poor-posture'}>
                <td>{row.id}</td>
                <td>{row.timestamp}</td>
                <td>{row.postureScore}%</td>
                <td>
                  <span className={`status-badge ${row.postureScore >= threshold ? 'good' : 'poor'}`}>
                    {row.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;
