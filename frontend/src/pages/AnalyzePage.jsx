import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'

const sleep = ms => new Promise(r => setTimeout(r, ms))

const FEATURE_LABELS = {
  url_length: 'URL length',
  hostname_length: 'Hostname length',
  path_length: 'Path length',
  num_dots: 'Dot count',
  num_hyphens: 'Hyphens in host',
  num_slashes: 'Slash count',
  num_underscores: 'Underscores',
  num_question_marks: 'Question marks',
  has_at_symbol: 'AT symbol (@)',
  num_subdomains: 'Subdomain depth',
  has_ip_address: 'IP in hostname',
  is_https: 'HTTPS scheme',
  is_url_shortener: 'URL shortener',
  is_punycode: 'Punycode / IDN',
  suspicious_keyword_count: 'Suspicious keywords',
  digit_ratio: 'Digit ratio',
  has_suspicious_tld: 'Suspicious TLD',
  double_slash_in_path: 'Double slash in path',
  min_brand_levenshtein: 'Brand similarity (Levenshtein)',
  sld_is_exact_brand: 'Known brand domain',
}

function isRisky(key, value) {
  const flaggedBooleans = ['has_at_symbol', 'has_ip_address', 'is_url_shortener', 'is_punycode', 'has_suspicious_tld', 'double_slash_in_path']
  if (flaggedBooleans.includes(key)) return value === 1
  if (key === 'is_https') return value === 0
  if (key === 'suspicious_keyword_count') return value > 0
  if (key === 'min_brand_levenshtein') return value >= 1 && value <= 3
  if (key === 'num_subdomains') return value > 2
  if (key === 'url_length') return value > 75
  if (key === 'digit_ratio') return value > 0.3
  if (key === 'num_slashes') return value > 4
  return false
}

function formatValue(key, value) {
  if (key === 'digit_ratio') return (value * 100).toFixed(1) + '%'
  if (typeof value === 'number' && value === 1 && ['has_at_symbol', 'has_ip_address', 'is_url_shortener', 'is_punycode', 'has_suspicious_tld', 'double_slash_in_path', 'is_https'].includes(key)) return 'Yes'
  if (typeof value === 'number' && value === 0 && ['has_at_symbol', 'has_ip_address', 'is_url_shortener', 'is_punycode', 'has_suspicious_tld', 'double_slash_in_path', 'is_https'].includes(key)) return 'No'
  if (key === 'min_brand_levenshtein' && value >= 99) return 'No match'
  return String(value)
}

function ScoreBar({ score }) {
  const pct = Math.round(score * 100)
  const color = score < 0.3 ? 'bg-emerald-500' : score < 0.55 ? 'bg-amber-500' : 'bg-red-500'
  const textColor = score < 0.3 ? 'text-emerald-600' : score < 0.55 ? 'text-amber-600' : 'text-red-600'
  return (
    <div className="mt-3">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs text-gray-500">Suspicion score</span>
        <span className={`text-sm font-mono font-bold ${textColor}`}>{pct}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function StageHeader({ number, title, subtitle, status }) {
  return (
    <div className="flex items-start gap-3">
      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 transition-colors duration-300 ${
        status === 'pending' ? 'bg-gray-100 text-gray-400' :
        status === 'running' ? 'bg-blue-100 text-blue-600' :
        'bg-emerald-100 text-emerald-600'
      }`}>
        {status === 'running' ? (
          <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
        ) : status === 'done' ? '✓' : number}
      </div>
      <div>
        <p className={`font-semibold text-sm transition-colors duration-300 ${
          status === 'pending' ? 'text-gray-400' : 'text-gray-800'
        }`}>{title}</p>
        <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>
      </div>
    </div>
  )
}

function VerdictCard({ verdict, mlAvailable }) {
  const config = {
    legitimate: {
      bg: 'bg-emerald-50 border-emerald-200',
      icon: '✓',
      iconBg: 'bg-emerald-100 text-emerald-600',
      title: 'Likely legitimate',
      desc: 'No significant phishing indicators detected in the URL.',
      textColor: 'text-emerald-800',
    },
    suspicious: {
      bg: 'bg-amber-50 border-amber-200',
      icon: '!',
      iconBg: 'bg-amber-100 text-amber-600',
      title: 'Suspicious',
      desc: 'Several phishing indicators detected. Proceed with caution.',
      textColor: 'text-amber-800',
    },
    phishing: {
      bg: 'bg-red-50 border-red-200',
      icon: '✕',
      iconBg: 'bg-red-100 text-red-600',
      title: 'Likely phishing',
      desc: 'Multiple strong phishing indicators detected. Do not interact with this URL.',
      textColor: 'text-red-800',
    },
  }
  const c = config[verdict]
  const note = mlAvailable
    ? 'Based on lexical + ML analysis — visual module pending.'
    : 'Based on lexical analysis only — ML and visual modules pending.'
  return (
    <div className={`rounded-xl border-2 p-5 flex items-start gap-4 ${c.bg}`}>
      <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold flex-shrink-0 ${c.iconBg}`}>
        {c.icon}
      </div>
      <div>
        <p className={`font-bold text-lg ${c.textColor}`}>{c.title}</p>
        <p className={`text-sm mt-0.5 ${c.textColor} opacity-75`}>{c.desc}</p>
        <p className="text-xs text-gray-500 mt-2">{note}</p>
      </div>
    </div>
  )
}

export default function AnalyzePage() {
  const [url, setUrl] = useState('')
  const [phase, setPhase] = useState('idle')
  const [error, setError] = useState('')

  const [lexical, setLexical] = useState({ status: 'pending', score: null, visible: [] })
  const [ml, setMl] = useState({ status: 'pending', score: null, note: '' })
  const [visual, setVisual] = useState({ status: 'pending', brand: null, note: '' })
  const [verdict, setVerdict] = useState(null)
  const [mlAvailable, setMlAvailable] = useState(false)

  const navigate = useNavigate()

  const reset = () => {
    setPhase('idle')
    setError('')
    setLexical({ status: 'pending', score: null, visible: [] })
    setMl({ status: 'pending', score: null, note: '' })
    setVisual({ status: 'pending', brand: null, note: '' })
    setVerdict(null)
    setMlAvailable(false)
  }

  const handleAnalyze = useCallback(async () => {
    if (!url.trim()) return
    reset()
    setPhase('running')

    // Stage 1 — Lexical
    setLexical({ status: 'running', score: null, visible: [] })
    let lexData
    try {
      const res = await api.post('/analyze/lexical', { url })
      lexData = res.data
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Check that the backend is running.')
      setPhase('error')
      return
    }

    const entries = Object.entries(lexData.features)
    for (let i = 0; i < entries.length; i++) {
      await sleep(90)
      setLexical(prev => ({ ...prev, visible: entries.slice(0, i + 1) }))
    }
    await sleep(200)
    setLexical({ status: 'done', score: lexData.score, visible: entries })

    await sleep(500)

    // Stage 2 — ML (Random Forest)
    setMl({ status: 'running', score: null, note: 'Loading Random Forest model...' })
    await sleep(400)
    setMl(prev => ({ ...prev, note: 'Running inference on 19 features...' }))
    let mlData = null
    try {
      const res = await api.post('/analyze/ml', { url })
      mlData = res.data
    } catch (err) {
      const detail = err.response?.data?.detail
      const status = err.response?.status
      if (status === 503) {
        setMl({ status: 'done', score: null, note: detail || 'ML model not trained yet.' })
      } else {
        setError(detail || 'ML analysis failed.')
        setPhase('error')
        return
      }
    }
    if (mlData) {
      await sleep(300)
      setMlAvailable(true)
      setMl({ status: 'done', score: mlData.score, note: null })
    }

    await sleep(400)

    // Stage 3 — Visual (placeholder)
    setVisual({ status: 'running', brand: null, note: 'Launching headless browser...' })
    await sleep(900)
    setVisual(prev => ({ ...prev, note: 'Capturing screenshot...' }))
    await sleep(800)
    setVisual(prev => ({ ...prev, note: 'Computing CLIP embedding...' }))
    await sleep(600)
    setVisual({ status: 'done', brand: null, note: 'Module not yet implemented — available after Day 9.' })

    // Combined verdict: average of available signals
    const signals = [lexData.score, mlData?.score].filter(s => typeof s === 'number')
    const s = signals.reduce((a, b) => a + b, 0) / Math.max(signals.length, 1)
    const v = s >= 0.55 ? 'phishing' : s >= 0.3 ? 'suspicious' : 'legitimate'
    setVerdict(v)
    setPhase('done')
  }, [url])

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/dashboard')} className="text-sm text-gray-400 hover:text-gray-600">← Dashboard</button>
          <span className="text-gray-300">|</span>
          <h1 className="text-lg font-bold text-gray-800">URL Analysis</h1>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto px-4 py-10">

        {/* URL input */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-bold text-gray-800 mb-1">Analyze a URL</h2>
          <p className="text-sm text-gray-500 mb-4">Enter any URL to run the three-stage phishing detection pipeline.</p>
          <div className="flex gap-2">
            <input
              type="text"
              value={url}
              onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && phase === 'idle' && handleAnalyze()}
              placeholder="https://example.com"
              disabled={phase === 'running'}
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-400"
            />
            {phase === 'idle' || phase === 'error' || phase === 'done' ? (
              <button
                onClick={phase === 'idle' ? handleAnalyze : reset}
                className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  phase === 'idle'
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {phase === 'idle' ? 'Analyze' : 'Reset'}
              </button>
            ) : (
              <button disabled className="px-5 py-2.5 rounded-lg text-sm font-medium bg-blue-100 text-blue-400 cursor-not-allowed">
                Running...
              </button>
            )}
          </div>
          {error && <p className="text-red-500 text-sm mt-3">{error}</p>}
        </div>

        {/* Stages */}
        {phase !== 'idle' && (
          <div className="space-y-4">

            {/* Stage 1 — Lexical */}
            <div className={`bg-white rounded-xl border transition-all duration-300 ${
              lexical.status === 'pending' ? 'border-gray-100 opacity-50' :
              lexical.status === 'running' ? 'border-blue-200 shadow-sm' :
              'border-gray-200 shadow-sm'
            } p-5`}>
              <StageHeader
                number="1"
                title="Lexical URL Analysis"
                subtitle="Pattern matching on URL string — no network requests"
                status={lexical.status}
              />

              {lexical.status !== 'pending' && (
                <div className="mt-4 ml-11">
                  <div className="flex flex-wrap gap-1.5">
                    {lexical.visible.map(([key, value]) => {
                      const risky = isRisky(key, value)
                      return (
                        <span
                          key={key}
                          className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono transition-all ${
                            risky
                              ? 'bg-red-50 text-red-700 border border-red-100'
                              : 'bg-gray-50 text-gray-600 border border-gray-100'
                          }`}
                        >
                          <span className="font-normal text-gray-400 hidden sm:inline">{FEATURE_LABELS[key] || key}:</span>
                          <span className="font-medium">{formatValue(key, value)}</span>
                          {risky && <span className="text-red-400">⚠</span>}
                        </span>
                      )
                    })}
                    {lexical.status === 'running' && (
                      <span className="inline-flex items-center px-2 py-1 rounded-md text-xs text-gray-400 bg-gray-50 border border-gray-100 animate-pulse">
                        scanning...
                      </span>
                    )}
                  </div>

                  {lexical.status === 'done' && lexical.score !== null && (
                    <ScoreBar score={lexical.score} />
                  )}
                </div>
              )}
            </div>

            {/* Stage 2 — ML */}
            <div className={`bg-white rounded-xl border transition-all duration-300 ${
              ml.status === 'pending' ? 'border-gray-100 opacity-40' :
              ml.status === 'running' ? 'border-blue-200 shadow-sm' :
              'border-gray-200 shadow-sm'
            } p-5`}>
              <StageHeader
                number="2"
                title="ML Classifier — Random Forest"
                subtitle="Trained on phishing URL corpus (Sahingoz et al., 2019 approach)"
                status={ml.status}
              />
              {ml.status !== 'pending' && (
                <div className="mt-3 ml-11">
                  {ml.note && (
                    <p className={`text-xs font-mono ${ml.status === 'running' ? 'text-blue-500' : 'text-gray-400'}`}>
                      {ml.note}
                    </p>
                  )}
                  {ml.status === 'done' && ml.score !== null && <ScoreBar score={ml.score} />}
                </div>
              )}
            </div>

            {/* Stage 3 — Visual */}
            <div className={`bg-white rounded-xl border transition-all duration-300 ${
              visual.status === 'pending' ? 'border-gray-100 opacity-40' :
              visual.status === 'running' ? 'border-blue-200 shadow-sm' :
              'border-gray-200 shadow-sm'
            } p-5`}>
              <StageHeader
                number="3"
                title="Visual Brand Matching"
                subtitle="CLIP embeddings + cosine similarity against 50-brand knowledge base"
                status={visual.status}
              />
              {visual.status !== 'pending' && (
                <div className="mt-3 ml-11">
                  <p className={`text-xs font-mono ${visual.status === 'running' ? 'text-blue-500' : 'text-gray-400'}`}>
                    {visual.note}
                  </p>
                  {visual.status === 'done' && visual.brand && (
                    <div className="mt-2 text-sm text-gray-700">
                      Matched brand: <span className="font-semibold">{visual.brand}</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Verdict */}
            {verdict && (
              <div className="pt-2">
                <VerdictCard verdict={verdict} mlAvailable={mlAvailable} />
              </div>
            )}

          </div>
        )}
      </main>
    </div>
  )
}