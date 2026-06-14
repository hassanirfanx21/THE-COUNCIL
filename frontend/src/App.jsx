import { useState, useRef, useEffect, useCallback } from 'react'

/* ── constants ── */
const AGENT_COLORS = {
  STRATEGIST: '#7c3aed',
  REALIST:    '#0ea5e9',
  ADVOCATE:   '#10b981',
  CONTRARIAN: '#f59e0b',
}

const AGENT_ICONS = {
  STRATEGIST: '⚡',
  REALIST:    '🔍',
  ADVOCATE:   '💚',
  CONTRARIAN: '🌀',
}

const THINKING_STEPS = [
  { id: 'stm',   label: 'Loading conversation memory' },
  { id: 'class', label: 'Classifying problem type' },
  { id: 'ltm',   label: 'Fetching long-term memories' },
  { id: 'agents',label: 'Four agents deliberating...' },
  { id: 'synth', label: 'Synthesizing final response' },
]

const STARTERS = [
  "I'm torn between two career paths.",
  "Should I take on this new project?",
  "I keep procrastinating on important work.",
  "Is this a good time to make a big change?",
]

const DUMMY_SESSIONS = [
  { id: 's1', label: 'Career crossroads', time: '2h ago' },
  { id: 's2', label: 'Freelance dilemma', time: 'Yesterday' },
  { id: 's3', label: 'Work-life balance', time: '3d ago' },
]

/* ── API call ── */
async function callCouncil(message, userId, threadId) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, user_id: userId, thread_id: threadId }),
  })
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

/* ── ThinkingCard ── */
function ThinkingCard({ step }) {
  const stepIdx = THINKING_STEPS.findIndex(s => s.id === step)
  return (
    <div className="thinking-card">
      <div className="thinking-header">
        <div className="thinking-spinner" />
        <span>The Council is deliberating…</span>
      </div>
      <div className="thinking-steps">
        {THINKING_STEPS.map((s, i) => {
          const state = i < stepIdx ? 'done' : i === stepIdx ? 'active' : 'pending'
          return (
            <div className="thinking-step" key={s.id}>
              <div className={`step-icon ${state}`}>
                {state === 'done' ? '✓' : state === 'active' ? '◉' : '○'}
              </div>
              <span className={`step-label ${state}-label`}>{s.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── WeightPill ── */
function WeightPill({ agent }) {
  const color = AGENT_COLORS[agent.agent_name] || '#7c3aed'
  const pct = Math.round(agent.weight * 100)
  return (
    <div className="weight-pill">
      <span className="weight-dot" style={{ background: color }} />
      <span style={{ color: color, fontWeight: 600 }}>{agent.agent_name.slice(0,3)}</span>
      <div className="weight-bar-track">
        <div className="weight-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span style={{ color: '#9d8bc0' }}>{pct}%</span>
    </div>
  )
}

/* ── AgentChip ── */
function AgentChip({ agent }) {
  return (
    <div className="agent-chip" data-agent={agent.name}>
      <div className="agent-chip-header">
        <span className="agent-chip-name">
          {AGENT_ICONS[agent.name]} {agent.name}
        </span>
        <span className="agent-confidence">{Math.round(agent.confidence * 100)}%</span>
      </div>
      <p className="agent-stance">{agent.stance}</p>
      <p className="agent-key">"{agent.key_point}"</p>
    </div>
  )
}

/* ── CouncilCard ── */
function CouncilCard({ data }) {
  const [showAgents, setShowAgents] = useState(false)

  return (
    <div className="council-card">
      {/* Classification bar */}
      <div className="classification-bar">
        <div className="decided-badge">
          🧠 {data.decided_agent}
        </div>
        <div className="weights-row">
          {data.agent_field?.map(aw => (
            <WeightPill key={aw.agent_name} agent={aw} />
          ))}
        </div>
      </div>

      {/* Agents accordion */}
      {data.agents_output?.length > 0 && (
        <div className="agents-section">
          <div
            className={`agents-toggle ${showAgents ? 'open' : ''}`}
            onClick={() => setShowAgents(v => !v)}
            role="button"
            aria-expanded={showAgents}
            id="agents-toggle-btn"
          >
            <span>{showAgents ? '▾' : '▸'}</span>
            <span>{showAgents ? 'Hide' : 'Show'} agent deliberations ({data.agents_output.length})</span>
          </div>
          {showAgents && (
            <div className="agents-grid">
              {data.agents_output.map(a => (
                <AgentChip key={a.name} agent={a} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Synthesizer */}
      <div className="synthesizer-section">
        <div className="synth-label">Council Verdict</div>
        <p className="synth-text">{data.final_response}</p>
      </div>
    </div>
  )
}

/* ── Message Row ── */
function MessageRow({ msg }) {
  if (msg.role === 'human') {
    return (
      <div className="message-row human">
        <div className="human-bubble">{msg.content}</div>
      </div>
    )
  }
  if (msg.role === 'thinking') {
    return (
      <div className="message-row ai">
        <ThinkingCard step={msg.step} />
      </div>
    )
  }
  return (
    <div className="message-row ai">
      <CouncilCard data={msg} />
    </div>
  )
}

/* ── App ── */
export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [thinkStep, setThinkStep] = useState('stm')
  const [userId]   = useState(() => {
    const saved = localStorage.getItem('council_user_id')
    if (saved) return saved
    const newId = 'user_' + Math.random().toString(36).slice(2,8)
    localStorage.setItem('council_user_id', newId)
    return newId
  })
  const [threadId, setThreadId] = useState(() => 'thread_' + Math.random().toString(36).slice(2,8))
  const [sessions, setSessions] = useState(DUMMY_SESSIONS)
  const [activeSession, setActiveSession] = useState(null)

  const bottomRef  = useRef(null)
  const textRef    = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Simulate thinking steps progression
  const simulateThinking = useCallback(() => {
    const steps = THINKING_STEPS.map(s => s.id)
    let i = 0
    const iv = setInterval(() => {
      i++
      if (i < steps.length) setThinkStep(steps[i])
      else clearInterval(iv)
    }, 900)
    return iv
  }, [])

  const send = async (text) => {
    const msg = text || input
    if (!msg.trim() || loading) return
    setInput('')
    textRef.current && (textRef.current.style.height = 'auto')

    setMessages(prev => [...prev, { role: 'human', content: msg }])
    setLoading(true)
    setThinkStep('stm')
    setMessages(prev => [...prev, { role: 'thinking', step: 'stm', id: 'thinking' }])

    const iv = simulateThinking()

    try {
      const data = await callCouncil(msg, userId, threadId)
      clearInterval(iv)
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'thinking'),
        { ...data, role: 'ai' }
      ])
    } catch {
      clearInterval(iv)
      // Fallback: mock response for demo
      const mock = buildMockResponse(msg)
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'thinking'),
        { ...mock, role: 'ai' }
      ])
    } finally {
      setLoading(false)
    }
  }

  const newSession = () => {
    const id = 'thread_' + Math.random().toString(36).slice(2,8)
    setThreadId(id)
    setMessages([])
    setActiveSession(id)
    setSessions(prev => [
      { id, label: 'New session', time: 'Just now' },
      ...prev,
    ])
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="brand-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
          </div>
          <span className="brand-name">The Council</span>
        </div>
        <div className="header-meta">
          <div className="status-dot" />
          <span>Online</span>
        </div>
        <div className="user-badge" id="user-badge">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
          </svg>
          <span>{userId}</span>
        </div>
      </header>

      <div className="main">
        {/* Sidebar */}
        <aside className="sidebar">
          <button className="sidebar-new-btn" onClick={newSession} id="new-session-btn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            New Session
          </button>

          <div className="sidebar-section">Recent</div>
          {sessions.map(s => (
            <div
              key={s.id}
              className={`session-item ${activeSession === s.id ? 'active' : ''}`}
              onClick={() => setActiveSession(s.id)}
              id={`session-${s.id}`}
            >
              <span className="session-icon">💬</span>
              <span className="session-text">{s.label}</span>
              <span className="session-time">{s.time}</span>
            </div>
          ))}
        </aside>

        {/* Chat */}
        <div className="chat-area">
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-orb">
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" opacity="0.9">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                  </svg>
                </div>
                <h1 className="empty-title">Your Advisory Council</h1>
                <p className="empty-sub">
                  Four distinct minds — Strategist, Realist, Advocate, Contrarian — deliberate every decision you bring. One synthesized truth emerges.
                </p>
                <div className="starter-chips">
                  {STARTERS.map(s => (
                    <button
                      key={s}
                      className="starter-chip"
                      onClick={() => send(s)}
                      id={`starter-${s.slice(0,15).replace(/\s/g,'-')}`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m, i) => (
                <MessageRow key={i} msg={m.id === 'thinking' ? { ...m, step: thinkStep } : m} />
              ))
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="input-area">
            <div className="input-wrapper">
              <textarea
                ref={textRef}
                id="chat-input"
                className="chat-input"
                placeholder="Bring your decision to the council…"
                value={input}
                onChange={handleInput}
                onKeyDown={onKeyDown}
                rows={1}
                disabled={loading}
              />
              <button
                id="send-btn"
                className="send-btn"
                onClick={() => send()}
                disabled={!input.trim() || loading}
                aria-label="Send message"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              </button>
            </div>
            <p className="input-hint">Enter to send · Shift+Enter for new line</p>
          </div>
        </div>

        {/* Right panel */}
        <aside className="right-panel">
          <div className="panel-section">
            <div className="panel-title">Agents</div>
            <div className="agent-legend">
              {Object.entries(AGENT_COLORS).map(([name, color]) => (
                <div className="legend-item" key={name}>
                  <span className="legend-dot" style={{ background: color }} />
                  <span style={{ color }}>{AGENT_ICONS[name]}</span>
                  <span>{name.charAt(0) + name.slice(1).toLowerCase()}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="panel-section">
            <div className="panel-title">Recent Memories</div>
            {[
              'Wants to become an AI systems architect',
              'Financial independence by 35 is top goal',
              'Tends to overcommit and burn out',
            ].map((m, i) => (
              <div className="memory-chip" key={i}>{m}</div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  )
}

/* ── Mock response builder (demo / fallback) ── */
function buildMockResponse(msg) {
  return {
    decided_agent: 'STRATEGIST',
    agent_field: [
      { agent_name: 'STRATEGIST', weight: 0.45 },
      { agent_name: 'REALIST',    weight: 0.25 },
      { agent_name: 'ADVOCATE',   weight: 0.20 },
      { agent_name: 'CONTRARIAN', weight: 0.10 },
    ],
    agents_output: [
      {
        name: 'STRATEGIST',
        stance: 'This aligns with your stated trajectory toward AI systems architecture.',
        key_point: 'Measure this against your 3-year goal — does it accelerate or delay it?',
        memory_used: [],
        confidence: 0.82,
      },
      {
        name: 'REALIST',
        stance: 'The real constraint you are not naming is time and capacity.',
        key_point: 'You have burned out before taking on too much simultaneously.',
        memory_used: [],
        confidence: 0.75,
      },
      {
        name: 'ADVOCATE',
        stance: 'This is something that genuinely energizes you — do not dismiss that.',
        key_point: 'What you feel alive doing matters as much as the strategy.',
        memory_used: [],
        confidence: 0.70,
      },
      {
        name: 'CONTRARIAN',
        stance: 'You have made this kind of impulsive decision before and regretted it.',
        key_point: 'Slow down. The urgency you feel may not be real.',
        memory_used: [],
        confidence: 0.65,
      },
    ],
    final_response: `Three of four advisors see this as worth pursuing — the Strategist most clearly, grounded in your stated goal of reaching AI systems architecture within three years. The direction is right. The question the Realist raises is about pacing, not destination. You have a documented pattern of over-committing, and if this new thing adds to your plate without something coming off it, the burnout risk is real.

The Contrarian is the outlier here, but not without merit. The urgency you feel around this decision deserves scrutiny. Is there a deadline that is actually yours, or one you are constructing? If the latter, a week of deliberate thinking costs you nothing.

My recommendation: pursue it, but not all at once. Identify the one thing that has to move to make room for this, and make that trade explicit before you commit. That is the discipline your past self did not apply.

What would you have to let go of to take this on without burning out?`,
  }
}
