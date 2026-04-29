import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState<"chat" | "predict">("chat");
  const [question, setQuestion] = useState("");
  const [chatLog, setChatLog] = useState<{ role: "user" | "bot"; text: string }[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  // All Original Params Included
  const [form, setForm] = useState({
    age: 30,
    sex: 1,
    bmi: 24,
    children: 0,
    smoker: 0,
    disease: 5,
    policy_type: 1
  });

  const [premium, setPremium] = useState<number | null>(null);

  // Auto-scroll to bottom when new messages arrive or loading state changes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatLog, isLoading]);

  const askBot = async () => {
    if (!question.trim()) return;
    const userQ = question;
    setChatLog(prev => [...prev, { role: "user", text: userQ }]);
    setQuestion("");
    setIsLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/chat", { question: userQ });
      setChatLog(prev => [...prev, { role: "bot", text: res.data.answer }]);
    } catch (error) {
      setChatLog(prev => [...prev, { role: "bot", text: "Sorry, I couldn't process your request." }]);
    }
    setIsLoading(false);
  };

  const predictPremium = async () => {
    setIsLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", form);
      setPremium(res.data.premium);
    } catch (error) {
      alert("Failed to calculate premium. Please try again.");
    }
    setIsLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askBot();
    }
  };

  return (
    <div className="layout">
      <div className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">🛡️</div>
          <h2>Shield AI</h2>
          <p className="tagline">Smart Insurance Solutions</p>
        </div>
        
        <nav className="nav-section">
          <button 
            className={`nav-btn ${activeTab === "chat" ? "active" : ""}`} 
            onClick={() => setActiveTab("chat")}
          >
            <span className="nav-icon">✨</span>
            <span className="nav-text">AI Assistant</span>
          </button>
          <button 
            className={`nav-btn ${activeTab === "predict" ? "active" : ""}`} 
            onClick={() => setActiveTab("predict")}
          >
            <span className="nav-icon">💳</span>
            <span className="nav-text">Premium Calculator</span>
          </button>
        </nav>
      </div>

      <div className="main-content">
        {activeTab === "chat" && (
          <div className="elegant-card chat-card">
            <div className="card-header">
              <div>
                <h2>Insurance Assistant</h2>
                <p className="subtitle">Ask me anything about your insurance policy</p>
              </div>
              <div className="status-badge">
                <span className="status-dot"></span>
                Online
              </div>
            </div>
            
            <div className="chat-window">
              {chatLog.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon">💡</div>
                  <h3>How can I help you today?</h3>
                  <p>Ask me about policy coverage, claims, premiums, or any insurance-related questions.</p>
                </div>
              )}
              {chatLog.map((msg, i) => (
                <div key={i} className={`message-wrapper ${msg.role}`}>
                  <div className={`message ${msg.role}`}>
                    {msg.role === "bot" && <div className="bot-avatar">🤖</div>}
                    <div className="message-content">{msg.text}</div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message-wrapper bot">
                  <div className="message bot">
                    <div className="bot-avatar">🤖</div>
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            
            <div className="chat-input-container">
              <input 
                className="chat-input"
                value={question} 
                onChange={e => setQuestion(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message here..."
                disabled={isLoading}
              />
              <button 
                className="send-btn" 
                onClick={askBot}
                disabled={!question.trim() || isLoading}
              >
                <span>Send</span>
                <span className="send-icon">→</span>
              </button>
            </div>
          </div>
        )}

        {activeTab === "predict" && (
          <div className="elegant-card predict-card">
            <div className="card-header">
              <div>
                <h2>Premium Calculator</h2>
                <p className="subtitle">Get an instant estimate of your insurance premium</p>
              </div>
            </div>
            
            <div className="form-container">
              <div className="form-section">
                <h3 className="section-title">Personal Information</h3>
                <div className="form-grid">
                  <div className="input-field">
                    <label>Age <span className="required">*</span></label>
                    <input 
                      type="number" 
                      value={form.age} 
                      onChange={e => setForm({...form, age: +e.target.value})}
                      min="18"
                      max="100"
                    />
                  </div>
                  
                  <div className="input-field">
                    <label>Gender <span className="required">*</span></label>
                    <select value={form.sex} onChange={e => setForm({...form, sex: +e.target.value})}>
                      <option value={1}>Male</option>
                      <option value={0}>Female</option>
                    </select>
                  </div>
                  
                  <div className="input-field">
                    <label>BMI (Body Mass Index) <span className="required">*</span></label>
                    <input 
                      type="number" 
                      value={form.bmi} 
                      onChange={e => setForm({...form, bmi: +e.target.value})}
                      step="0.1"
                      min="10"
                      max="50"
                    />
                    <span className="input-hint">Normal range: 18.5 - 24.9</span>
                  </div>
                  
                  <div className="input-field">
                    <label>Number of Children</label>
                    <input 
                      type="number" 
                      value={form.children} 
                      onChange={e => setForm({...form, children: +e.target.value})}
                      min="0"
                      max="10"
                    />
                  </div>
                </div>
              </div>

              <div className="form-section">
                <h3 className="section-title">Health & Lifestyle</h3>
                <div className="form-grid">
                  <div className="input-field">
                    <label>Smoking Status <span className="required">*</span></label>
                    <select value={form.smoker} onChange={e => setForm({...form, smoker: +e.target.value})}>
                      <option value={0}>Non-Smoker</option>
                      <option value={1}>Smoker</option>
                    </select>
                  </div>
                  
                  <div className="input-field">
                    <label>Pre-existing Condition <span className="required">*</span></label>
                    <select value={form.disease} onChange={e => setForm({...form, disease: +e.target.value})}>
                      <option value={5}>None</option>
                      <option value={1}>Diabetes</option>
                      <option value={3}>Hypertension</option>
                      <option value={2}>Heart Disease</option>
                      <option value={0}>Asthma</option>
                      <option value={4}>Multiple Conditions</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="form-section">
                <h3 className="section-title">Policy Details</h3>
                <div className="form-grid">
                  <div className="input-field full-width">
                    <label>Policy Type <span className="required">*</span></label>
                    <select value={form.policy_type} onChange={e => setForm({...form, policy_type: +e.target.value})}>
                      <option value={1}>Individual</option>
                      <option value={0}>Family Floater</option>
                      <option value={2}>Senior Citizen</option>
                    </select>
                  </div>
                </div>
              </div>
              
              <button 
                className="predict-btn" 
                onClick={predictPremium}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="btn-spinner"></span>
                    Calculating...
                  </>
                ) : (
                  <>
                    <span>Calculate Premium</span>
                    <span className="btn-icon"></span>
                  </>
                )}
              </button>
            </div>

            {premium !== null && (
              <div className="result-banner">
                <div className="result-header">
                  <div className="result-icon">✨</div>
                  <div>
                    <p className="result-label">Estimated Annual Premium</p>
                    <h1 className="result-amount">₹{premium.toLocaleString('en-IN')}</h1>
                    <p className="result-note">*This is an estimate. Final premium may vary based on additional factors.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;