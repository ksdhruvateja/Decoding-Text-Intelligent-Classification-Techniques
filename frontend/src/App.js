import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Send, Trash2, AlertCircle, Sparkles, Activity, ShieldCheck } from 'lucide-react';
import './App.css';

function App() {
  const [messageInput, setMessageInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [availableCategories, setAvailableCategories] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const chatContainerRef = useRef(null);
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  const fetchAvailableCategories = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/categories`);
      const data = await response.json();
      setAvailableCategories(data.categories || []);
    } catch (error) {
      console.error('Error fetching categories:', error);
    }
  }, [API_BASE_URL]);

  const loadHistoryFromServer = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history?limit=50`);
      const data = await response.json();
      setChatHistory(data.history || []);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchAvailableCategories();
    loadHistoryFromServer();
  }, [fetchAvailableCategories, loadHistoryFromServer]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const classifyTextMessage = async (textContent) => {
    setIsProcessing(true);
    setErrorMessage('');

    try {
      // Request both JSON and formatted block for display
      const jsonRespPromise = fetch(`${API_BASE_URL}/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textContent,
          threshold: 0.5,
        }),
      });
      const blockRespPromise = fetch(`${API_BASE_URL}/classify-formatted`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textContent }),
      });

      const [jsonResp, blockResp] = await Promise.all([jsonRespPromise, blockRespPromise]);

      if (!jsonResp.ok) {
        throw new Error('Classification request failed');
      }
      const classificationData = await jsonResp.json();

      let formattedBlock = '';
      if (blockResp.ok) {
        formattedBlock = await blockResp.text();
      }

      const merged = { ...classificationData, formatted_block: formattedBlock };
      setChatHistory(prev => [...prev, merged]);
      return merged;
    } catch (error) {
      setErrorMessage('Failed to classify text. Please try again.');
      console.error('Classification error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!messageInput.trim() || isProcessing) {
      return;
    }

    const textToClassify = messageInput.trim();
    setMessageInput('');

    await classifyTextMessage(textToClassify);
  };

  const clearChatHistory = async () => {
    try {
      setErrorMessage('');
      setSuccessMessage('');
      
      const response = await fetch(`${API_BASE_URL}/history/clear`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to clear history');
      }
      
      const data = await response.json();
      setChatHistory([]);
      setErrorMessage('');
      setSuccessMessage('Telemetry log cleared successfully');
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(''), 3000);
      
      console.log('History cleared:', data.message);
    } catch (error) {
      console.error('Error clearing history:', error);
      setErrorMessage('Failed to clear telemetry log. Please try again.');
      setSuccessMessage('');
    }
  };

  const getCategoryColor = (categoryName) => {
    const colorMap = {
      toxic: '#ef4444',
      severe_toxic: '#dc2626',
      obscene: '#f97316',
      threat: '#ea580c',
      insult: '#eab308',
      identity_hate: '#dc2626',
    };
    return colorMap[categoryName] || '#6366f1';
  };

  const insightMetrics = useMemo(() => {
    const totalDetections = chatHistory.length;
    const criticalAlerts = chatHistory.filter(item => item.sentiment && item.sentiment !== 'safe').length;
    const lastEmotion = chatHistory[chatHistory.length - 1]?.emotion || 'neutral';

    return {
      totalDetections,
      criticalAlerts,
      lastEmotion,
    };
  }, [chatHistory]);

  const latestResult = chatHistory[chatHistory.length - 1] || null;

  const pipelineStages = useMemo(() => {
    const stageStatus = (opts = {}) => {
      if (isProcessing) return 'active';
      if (!latestResult) return 'idle';
      if (opts.alertCondition) return 'alert';
      return 'complete';
    };

    const formatPercent = (value) => `${Math.round((value ?? 0) * 100)}%`;

    return [
      {
        id: 'stage-simple',
        step: 'Tier 1',
        title: 'SimpleClassifier (Rule-Based)',
        detail: latestResult
          ? `Pattern matching complete: ${latestResult.primary_category} (${formatPercent(latestResult.confidence)}) | Emotion: ${latestResult.emotion || 'neutral'}`
          : 'Fast pattern matching with 200+ keywords and 30+ threat patterns. Ready to scan.',
        status: stageStatus(),
      },
      {
        id: 'stage-bert',
        step: 'Tier 2',
        title: 'BERT Classifier (Deep Learning)',
        detail: latestResult
          ? `BERT inference: bert-base-uncased (110M params) analyzed semantic context with ${Object.keys(latestResult.all_scores || {}).length} categories.`
          : 'Neural network idle. Semantic understanding via 12 transformer layers (768-dim embeddings).',
        status: stageStatus(),
      },
      {
        id: 'stage-tfidf',
        step: 'Tier 3',
        title: 'TF-IDF Classifier (ML Fallback)',
        detail: latestResult
          ? `Logistic regression fallback ready. Primary decision: ${latestResult.primary_category} | Confidence: ${formatPercent(latestResult.confidence)}`
          : 'Logistic regression with term frequency analysis. 76.4% accuracy fallback ready.',
        status: stageStatus(),
      },
      {
        id: 'stage-blend',
        step: 'Orchestrator',
        title: 'ProductionClassifier (Hybrid Blending)',
        detail: latestResult
          ? latestResult.confidence >= 0.80 && ['self_harm_high', 'self_harm_low', 'unsafe_environment'].includes(latestResult.primary_category)
            ? `HIGH-RISK OVERRIDE: ${latestResult.primary_category} detected with ${formatPercent(latestResult.confidence)} confidence. Rule-based decision enforced.`
            : `Weighted blend (75% BERT, 25% rule-based) with safety overrides. Final: ${latestResult.primary_category}`
          : 'Intelligent orchestration: blends all classifiers with safety-first overrides for high-risk content.',
        status: stageStatus({ alertCondition: latestResult?.confidence >= 0.80 && ['self_harm_high', 'self_harm_low', 'unsafe_environment'].includes(latestResult?.primary_category) }),
      },
      {
        id: 'stage-api',
        step: 'Output',
        title: 'Flask API Response',
        detail: latestResult
          ? `Normalized scores (sum=100.0%), sentiment=${latestResult.sentiment}, stored at ${new Date(latestResult.timestamp).toLocaleTimeString()}`
          : 'RESTful API endpoint /api/classify ready. Returns JSON with category, confidence, sentiment, emotion.',
        status: stageStatus(),
      },
    ];
  }, [latestResult, isProcessing]);

  return (
    <div className="app-shell">
      <div className="neon-orbit neon-orbit--one" aria-hidden="true" />
      <div className="neon-orbit neon-orbit--two" aria-hidden="true" />

      <header className="hero">
        <div className="hero-content">
              <p className="hero-kicker">Group 5 • Decoding Text : Intelligent Classification Techniques</p>
              <h1>Decoding Text : Intelligent Classification Techniques</h1>
              <p className="hero-subtitle">
                Intelligent multi-label text classification with advanced BERT pipelines, LLM verification, and comprehensive analysis.
              </p>
          <div className="hero-badges">
            <div className="data-badge">
              <Sparkles size={20} />
              <div>
                <small>Detections</small>
                <strong>{insightMetrics.totalDetections}</strong>
              </div>
            </div>
            <div className="data-badge">
              <Activity size={20} />
              <div>
                <small>Critical Alerts</small>
                <strong>{insightMetrics.criticalAlerts}</strong>
              </div>
            </div>
            <div className="data-badge">
              <ShieldCheck size={20} />
              <div>
                <small>Latest Emotion</small>
                <strong>{insightMetrics.lastEmotion}</strong>
              </div>
            </div>
          </div>
        </div>

            <div className="team-card holo-card">
              <p className="team-label">Project By</p>
              <h3>Dhruva Teja Kandalam Sunil</h3>
              <span className="team-meta">SBU ID · 116451406</span>
            </div>
      </header>

      <div className="layout-grid">
        <aside className="control-dock">
          <div className="holo-card">
            <div className="control-headline">
              <span>Signal Channels</span>
              <small>Live feed from classifier core</small>
            </div>
            <div className="category-grid">
              {availableCategories.map((category, idx) => (
                <div
                  key={idx}
                  className="category-chip"
                  style={{ borderColor: getCategoryColor(category) }}
                >
                  <span className="chip-dot" style={{ backgroundColor: getCategoryColor(category) }} />
                  {category.replace('_', ' ')}
                </div>
              ))}
            </div>
          </div>

          <div className="control-actions holo-card">
            <p>System Controls</p>
            <button
              onClick={clearChatHistory}
              className="btn btn-danger"
            >
              <Trash2 size={16} />
              Clear telemetry log
            </button>
          </div>
        </aside>

        <div className="analysis-stack">
          <main className="console-panel">
            {errorMessage && (
              <div className="alert-banner">
                <AlertCircle size={20} />
                <span>{errorMessage}</span>
              </div>
            )}
            {successMessage && (
              <div className="alert-banner" style={{ backgroundColor: '#10b981', borderColor: '#059669' }}>
                <span>{successMessage}</span>
              </div>
            )}

            <div className="chat-stream" ref={chatContainerRef}>
              {chatHistory.length === 0 ? (
                <div className="empty-state">
                  <h3>Initiate transmission</h3>
                  <p>Send any phrase below to witness the holographic classifier in action.</p>
                </div>
              ) : (
                chatHistory.map((item, index) => (
                  <div key={index} className="message-card neo-card">
                    <div className="message-meta">
                      <span className="meta-time">
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </span>
                    {(() => {
                      const highRiskCategories = ['self_harm_high', 'self_harm_low', 'unsafe_environment'];
                      const isHighRisk = highRiskCategories.includes(item.primary_category);
                      const safeStates = ['safe', 'neutral', 'positive'];
                      const isSafe = safeStates.includes(item.sentiment) && !isHighRisk;
                      const statusClass = isSafe
                        ? 'meta-status--safe'
                        : isHighRisk
                          ? 'meta-status--alert'
                          : 'meta-status--warn';
                      const statusLabel = isSafe
                        ? 'Safe channel'
                        : isHighRisk
                          ? 'Critical alert'
                          : 'Alert raised';
                      return (
                        <span className={`meta-status ${statusClass}`}>
                          {statusLabel}
                        </span>
                      );
                    })()}
                    </div>

                    <div className="message-body">
                      {item.text}
                    </div>

                    <div className="emotion-chip">
                      <span>Emotion vector</span>
                      <strong>{item.emotion || 'neutral'}</strong>
                    </div>

                    {item.formatted_block && (
                      <pre className="formatted-block" aria-label="Formatted classification block">
                        {item.formatted_block}
                      </pre>
                    )}

                    {item.predictions && item.predictions.length > 0 && (
                      <div className="prediction-stack">
                        <p className="stack-title">Detected labels</p>
                        <div className="stack-grid">
                          {item.predictions.map((pred, idx) => (
                            <div
                              key={idx}
                              className="prediction-pill"
                              style={{ borderColor: getCategoryColor(pred.label) }}
                            >
                              <span>{pred.label.replace('_', ' ')}</span>
                              <strong>{((pred.score || pred.confidence) * 100).toFixed(1)}%</strong>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {item.all_scores && (
                      <details className="scores-panel">
                        <summary>Confidence spectrum</summary>
                        <div className="spectrum-grid">
                          {Object.entries(item.all_scores).map(([label, score]) => (
                            <div key={label} className="spectrum-row">
                              <span>{label.replace('_', ' ')}</span>
                              <div className="spectrum-bar">
                                <div
                                  className="spectrum-fill"
                                  style={{
                                    width: `${score * 100}%`,
                                    backgroundColor: getCategoryColor(label),
                                  }}
                                />
                              </div>
                              <span>{(score * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                        {item.analysis_explanation && (
                          <details className="analysis-explanation">
                            <summary>Detailed Analysis</summary>
                            <div className="explanation-content">
                              {item.analysis_explanation.detected_patterns && item.analysis_explanation.detected_patterns.length > 0 && (
                                <div className="explanation-section">
                                  <h4>Detected Patterns</h4>
                                  <div className="pattern-tags">
                                    {item.analysis_explanation.detected_patterns.map((pattern, idx) => (
                                      <span key={idx} className="pattern-tag">
                                        {pattern.replace('_', ' ')}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {item.analysis_explanation.reasoning && item.analysis_explanation.reasoning.length > 0 && (
                                <div className="explanation-section">
                                  <h4>Reasoning</h4>
                                  <ul className="reasoning-list">
                                    {item.analysis_explanation.reasoning.map((reason, idx) => (
                                      <li key={idx}>{reason}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {item.analysis_explanation.key_indicators && item.analysis_explanation.key_indicators.length > 0 && (
                                <div className="explanation-section">
                                  <h4>Key Indicators</h4>
                                  <div className="indicator-tags">
                                    {item.analysis_explanation.key_indicators.slice(0, 5).map((indicator, idx) => (
                                      <span key={idx} className="indicator-tag">{indicator}</span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {item.analysis_explanation.similar_cases && item.analysis_explanation.similar_cases.length > 0 && (
                                <div className="explanation-section">
                                  <h4>Similar Cases</h4>
                                  <p className="similar-cases-text">
                                    {item.analysis_explanation.similar_cases.join(' • ')}
                                  </p>
                                </div>
                              )}
                              {item.analysis_explanation.sentiment_analysis && (
                                <div className="explanation-section">
                                  <h4>Sentiment Analysis</h4>
                                  <p className="sentiment-interpretation">
                                    <strong>Base Sentiment:</strong> {item.analysis_explanation.sentiment_analysis.base_sentiment}
                                    <br />
                                    <small>{item.analysis_explanation.sentiment_analysis.interpretation}</small>
                                  </p>
                                </div>
                              )}
                            </div>
                          </details>
                        )}
                    {item.analysis_explanation && (
                      <details className="analysis-explanation">
                        <summary>Detailed Analysis</summary>
                        <div className="explanation-content">
                          {item.analysis_explanation.detected_patterns && item.analysis_explanation.detected_patterns.length > 0 && (
                            <div className="explanation-section">
                              <h4>Detected Patterns</h4>
                              <div className="pattern-tags">
                                {item.analysis_explanation.detected_patterns.map((pattern, idx) => (
                                  <span key={idx} className="pattern-tag">
                                    {pattern.replace('_', ' ')}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {item.analysis_explanation.reasoning && item.analysis_explanation.reasoning.length > 0 && (
                            <div className="explanation-section">
                              <h4>Reasoning</h4>
                              <ul className="reasoning-list">
                                {item.analysis_explanation.reasoning.map((reason, idx) => (
                                  <li key={idx}>{reason}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {item.analysis_explanation.key_indicators && item.analysis_explanation.key_indicators.length > 0 && (
                            <div className="explanation-section">
                              <h4>Key Indicators</h4>
                              <div className="indicator-tags">
                                {item.analysis_explanation.key_indicators.slice(0, 5).map((indicator, idx) => (
                                  <span key={idx} className="indicator-tag">{indicator}</span>
                                ))}
                              </div>
                            </div>
                          )}
                          {item.analysis_explanation.similar_cases && item.analysis_explanation.similar_cases.length > 0 && (
                            <div className="explanation-section">
                              <h4>Similar Cases</h4>
                              <p className="similar-cases-text">
                                {item.analysis_explanation.similar_cases.join(' • ')}
                              </p>
                            </div>
                          )}
                          {item.analysis_explanation.sentiment_analysis && (
                            <div className="explanation-section">
                              <h4>Sentiment Analysis</h4>
                              <p className="sentiment-interpretation">
                                <strong>Base Sentiment:</strong> {item.analysis_explanation.sentiment_analysis.base_sentiment}
                                <br />
                                <small>{item.analysis_explanation.sentiment_analysis.interpretation}</small>
                              </p>
                            </div>
                          )}
                        </div>
                      </details>
                    )}
                    {item.llm_summary && item.llm_summary.adjustment && (
                      <div className="llm-insight">
                        <p className="llm-insight-title">LLM Insight</p>
                        <p className="llm-insight-rationale">{item.llm_summary.rationale}</p>
                        <span className={`llm-insight-adjustment llm-insight-adjustment--${item.llm_summary.adjustment}`}>
                          {item.llm_summary.adjustment.replace('_', ' ')}
                        </span>
                      </div>
                    )}
                    {item.analysis_details && (
                      <details className="analysis-details-panel">
                        <summary>Detailed Analysis</summary>
                        <div className="analysis-content">
                          {item.analysis_details.sentiment_analysis && (
                            <div className="analysis-section">
                              <h4>Sentiment Analysis</h4>
                              <p><strong>Detected:</strong> {item.analysis_details.sentiment_analysis.detected_sentiment}</p>
                              <p><strong>Confidence:</strong> {(item.analysis_details.sentiment_analysis.confidence * 100).toFixed(1)}%</p>
                              <p><strong>Explanation:</strong> {item.analysis_details.sentiment_analysis.explanation}</p>
                            </div>
                          )}
                          {item.analysis_details.detected_patterns && item.analysis_details.detected_patterns.length > 0 && (
                            <div className="analysis-section">
                              <h4>Detected Patterns</h4>
                              {item.analysis_details.detected_patterns.map((pattern, idx) => (
                                <div key={idx} className="pattern-item">
                                  <strong>{pattern.type.replace('_', ' ')}:</strong> {pattern.description}
                                  <br />
                                  <small>Impact: {pattern.impact}</small>
                                </div>
                              ))}
                            </div>
                          )}
                          {item.analysis_details.key_indicators && item.analysis_details.key_indicators.length > 0 && (
                            <div className="analysis-section">
                              <h4>Key Indicators</h4>
                              <div className="indicators-list">
                                {item.analysis_details.key_indicators.map((indicator, idx) => (
                                  <span key={idx} className="indicator-tag">{indicator}</span>
                                ))}
                              </div>
                            </div>
                          )}
                          {item.analysis_details.classification_reasoning && item.analysis_details.classification_reasoning.length > 0 && (
                            <div className="analysis-section">
                              <h4>Classification Reasoning</h4>
                              {item.analysis_details.classification_reasoning.map((reason, idx) => (
                                <div key={idx} className="reasoning-item">
                                  {reason.label && (
                                    <p><strong>{reason.label}:</strong> {((reason.score || reason.confidence) * 100).toFixed(1)}% confidence</p>
                                  )}
                                  {reason.explanation && <p>{reason.explanation}</p>}
                                  {reason.type && <p><em>Rule-based override applied</em></p>}
                                </div>
                              ))}
                            </div>
                          )}
                          {item.analysis_details.similar_cases && item.analysis_details.similar_cases.length > 0 && (
                            <div className="analysis-section">
                              <h4>Similar Cases</h4>
                              {item.analysis_details.similar_cases.map((similar, idx) => (
                                <div key={idx} className="similar-case">
                                  <strong>Pattern:</strong> {similar.pattern}
                                  <ul>
                                    {similar.examples.map((ex, exIdx) => (
                                      <li key={exIdx}>{ex}</li>
                                    ))}
                                  </ul>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </details>
                    )}
                  </div>
                ))
              )}
            </div>

            <form onSubmit={handleSendMessage} className="transmit-form">
              <div className="input-shell">
                <input
                  type="text"
                  value={messageInput}
                  onChange={(e) => setMessageInput(e.target.value)}
                  placeholder="Compose a transmission..."
                  className="message-input"
                  disabled={isProcessing}
                />
                <span className="input-glow" />
              </div>
              <button
                type="submit"
                className="send-button"
                disabled={isProcessing || !messageInput.trim()}
              >
                {isProcessing ? (
                  <span className="spinner" />
                ) : (
                  <>
                    <Send size={18} />
                    Launch
                  </>
                )}
              </button>
            </form>
          </main>

          <section className="backend-panel holo-card">
            <header className="backend-header">
              <div>
                <p className="backend-kicker">PyTorch · BERT · TF-IDF · Flask</p>
                <h3>Three-Tier Hybrid Pipeline</h3>
              </div>
              <span className={`backend-status ${isProcessing ? 'backend-status--active' : latestResult ? 'backend-status--online' : 'backend-status--idle'}`}>
                {isProcessing ? 'Processing' : latestResult ? 'Online' : 'Idle'}
              </span>
            </header>
            <p className="backend-description">
              Real-time view of the hybrid classification pipeline: SimpleClassifier (rule-based) → BERT (deep learning) → TF-IDF (ML fallback) → ProductionClassifier (intelligent blending with safety overrides).
            </p>
            <div className="pipeline-grid">
              {pipelineStages.map(stage => (
                <div key={stage.id} className={`pipeline-stage pipeline-stage--${stage.status}`}>
                  <div className="pipeline-meta">
                    <span className="stage-step">{stage.step}</span>
                    <span className="stage-status">{stage.status === 'active' ? 'Running' : stage.status === 'complete' ? 'Complete' : stage.status === 'alert' ? 'Override' : 'Idle'}</span>
                  </div>
                  <h4>{stage.title}</h4>
                  <p>{stage.detail}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;