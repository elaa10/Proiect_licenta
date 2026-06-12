import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'

const sleep = ms => new Promise(r => setTimeout(r, ms))

const FEATURE_LABELS = {
  url_length: 'URL length', hostname_length: 'Hostname length',
  path_length: 'Path length', num_dots: 'Dot count',
  num_hyphens: 'Hyphens in host', num_slashes: 'Slash count',
  num_underscores: 'Underscores', num_question_marks: 'Question marks',
  has_at_symbol: 'AT symbol (@)', num_subdomains: 'Subdomain depth',
  has_ip_address: 'IP in hostname', is_https: 'HTTPS scheme',
  is_url_shortener: 'URL shortener', is_punycode: 'Punycode / IDN',
  suspicious_keyword_count: 'Suspicious keywords', digit_ratio: 'Digit ratio',
  has_suspicious_tld: 'Suspicious TLD', double_slash_in_path: 'Double slash in path',
  min_brand_levenshtein: 'Brand similarity (Levenshtein)', sld_is_exact_brand: 'Known brand domain',
}

function isRisky(key, value) {
  const flagged = ['has_at_symbol', 'has_ip_address', 'is_url_shortener', 'is_punycode', 'has_suspicious_tld', 'double_slash_in_path']
  if (flagged.includes(key)) return value === 1
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
  const bools = ['has_at_symbol', 'has_ip_address', 'is_url_shortener', 'is_punycode', 'has_suspicious_tld', 'double_slash_in_path', 'is_https', 'sld_is_exact_brand']
  if (bools.includes(key)) return value === 1 ? 'Yes' : 'No'
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
        <div className={`h-full rounded-full transition-all duration-700 ${color}`} style={{ width: `${pct}%` }} />
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
        <p className={`font-semibold text-sm transition-colors duration-300 ${status === 'pending' ? 'text-gray-400' : 'text-gray-800'}`}>{title}</p>
        <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>
      </div>
    </div>
  )
}

function VerdictCard({ verdict, lexicalScore, mlScore, visualBrand }) {
  const config = {
    legitimate: {
      bg: 'bg-emerald-50 border-emerald-200', icon: '✓', iconBg: 'bg-emerald-100 text-emerald-600',
      title: 'Likely legitimate', desc: 'No significant phishing indicators detected.', textColor: 'text-emerald-800',
    },
    suspicious: {
      bg: 'bg-amber-50 border-amber-200', icon: '!', iconBg: 'bg-amber-100 text-amber-600',
      title: 'Suspicious', desc: 'Several phishing indicators detected. Proceed with caution.', textColor: 'text-amber-800',
    },
    phishing: {
      bg: 'bg-red-50 border-red-200', icon: '✕', iconBg: 'bg-red-100 text-red-600',
      title: 'Likely phishing', desc: 'Multiple strong phishing indicators detected. Do not interact with this URL.', textColor: 'text-red-800',
    },
  }
  const c = config[verdict] || config.suspicious
  const signals = ['lexical', mlScore !== null ? 'ML' : null, visualBrand !== null ? 'visual' : null].filter(Boolean)

  return (
    <div className={`rounded-xl border-2 p-5 flex items-start gap-4 ${c.bg}`}>
      <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold flex-shrink-0 ${c.iconBg}`}>{c.icon}</div>
      <div className="flex-1">
        <p className={`font-bold text-lg ${c.textColor}`}>{c.title}</p>
        <p className={`text-sm mt-0.5 ${c.textColor} opacity-75`}>{c.desc}</p>
        {visualBrand && verdict === 'legitimate' && (
          <p className="text-xs text-emerald-700 mt-1.5 font-medium">
            ✓ Visual module confirmed this is the official <span className="font-bold">{visualBrand}</span> page — ML suspicion score overridden.
          </p>
        )}
        {visualBrand && (verdict === 'suspicious' || verdict === 'phishing') && (
          <div className="mt-2 bg-white bg-opacity-60 rounded-lg px-3 py-2 border border-current border-opacity-20">
            <p className={`text-xs font-semibold ${c.textColor}`}>
              ⚠ Această pagină imită vizual <span className="font-bold">{visualBrand}</span>.
            </p>
            <p className={`text-xs mt-0.5 ${c.textColor} opacity-80`}>
              Verificați că URL-ul din bara de adrese este cel oficial al {visualBrand} înainte de a introduce date personale.
            </p>
          </div>
        )}
        {!visualBrand && (verdict === 'suspicious' || verdict === 'phishing') && (
          <p className={`text-xs mt-1.5 ${c.textColor} opacity-75`}>
            Niciun brand cunoscut nu a fost identificat vizual. Procedați cu precauție.
          </p>
        )}
        <p className="text-xs text-gray-500 mt-2">Based on {signals.join(' + ')} analysis.</p>
      </div>
    </div>
  )
}

export default function AnalyzePage() {
  const [url, setUrl] = useState('')
  const [visualModel, setVisualModel] = useState('clip')
  const [phase, setPhase] = useState('idle')
  const [error, setError] = useState('')
  const [stage1, setStage1] = useState({ status: 'pending', features: [] })
  const [stage2, setStage2] = useState({ status: 'pending', note: '' })
  const [stage3, setStage3] = useState({ status: 'pending', note: '' })
  const [result, setResult] = useState(null)
  const navigate = useNavigate()

  const reset = () => {
    setPhase('idle'); setError('')
    setStage1({ status: 'pending', features: [] })
    setStage2({ status: 'pending', note: '' })
    setStage3({ status: 'pending', note: '' })
    setResult(null)
  }

  const handleAnalyze = useCallback(async () => {
    if (!url.trim()) return
    reset()
    setPhase('running')

    // Stage 1 — Lexical
    setStage1({ status: 'running', features: [] })
    let lexData = null
    try {
      const res = await api.post('/analyze/lexical', { url })
      lexData = res.data
    } catch { }

    if (lexData) {
      const entries = Object.entries(lexData.features)
      for (let i = 0; i < entries.length; i++) {
        await sleep(80)
        setStage1(prev => ({ ...prev, features: entries.slice(0, i + 1) }))
      }
      setStage1({ status: 'done', score: lexData.score, features: entries })
    } else {
      setStage1({ status: 'done', score: null, features: [] })
    }
    await sleep(300)

    // Stage 2 — ML
    setStage2({ status: 'running', note: 'Running Random Forest inference...' })
    await sleep(800)
    setStage3({ status: 'running', note: 'Capturing screenshot...' })
    await sleep(1000)
    setStage3(prev => ({ ...prev, note: `Computing ${visualModel === 'clip' ? 'CLIP' : 'DINOv2'} embedding...` }))

    // Full pipeline call
    let data = null
    try {
      const res = await api.post('/analyze/', { url, visual_model: visualModel })
      data = res.data
    } catch (err) {
      // Fallback: try without visual_model param
      try {
        const res = await api.post('/analyze/', { url })
        data = res.data
      } catch (err2) {
        setError(err2.response?.data?.detail || 'Analysis failed.')
        setPhase('error')
        return
      }
    }

    setStage2({
      status: 'done',
      score: data.ml_score,
      note: data.ml_score === null ? 'ML model not available.' : null,
    })

    setStage3({
      status: 'done',
      note: data.screenshot_path ? null : 'Screenshot capture failed.',
      screenshotUrl: data.screenshot_path
        ? `http://localhost:8000/screenshots/${data.screenshot_path.split('/').pop()}`
        : null,
      matched: !!data.visual_match_brand,
      display: data.visual_match_brand,
      similarity: data.visual_similarity,
    })

    setResult(data)
    setPhase('done')
  }, [url, visualModel])

  const modelLabel = visualModel === 'clip' ? 'CLIP ViT-B/32' : 'DINOv2 ViT-B/14'

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
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-bold text-gray-800 mb-1">Analyze a URL</h2>
          <p className="text-sm text-gray-500 mb-4">Three-stage phishing detection: lexical · ML classifier · visual matching.</p>

          {/* Visual model selector */}
          <div className="flex items-center gap-3 mb-4 p-3 bg-gray-50 rounded-lg">
            <span className="text-xs text-gray-500 font-medium">Visual model:</span>
            <div className="flex gap-2">
              <button
                onClick={() => setVisualModel('clip')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  visualModel === 'clip'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-600 border border-gray-200 hover:border-blue-300'
                }`}
              >
                CLIP ViT-B/32
              </button>
              <button
                onClick={() => setVisualModel('dino')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  visualModel === 'dino'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white text-gray-600 border border-gray-200 hover:border-purple-300'
                }`}
              >
                DINOv2 ViT-B/14
              </button>
            </div>
          </div>

          <div className="flex gap-2">
            <input
              type="text" value={url} onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && phase === 'idle' && handleAnalyze()}
              placeholder="https://example.com" disabled={phase === 'running'}
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-400"
            />
            {phase === 'idle' || phase === 'error' || phase === 'done' ? (
              <button onClick={phase === 'idle' ? handleAnalyze : reset}
                className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  phase === 'idle' ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}>
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

        {phase !== 'idle' && (
          <div className="space-y-4">

            {/* Stage 1 */}
            <div className={`bg-white rounded-xl border transition-all duration-300 ${
              stage1.status === 'pending' ? 'border-gray-100 opacity-50' :
              stage1.status === 'running' ? 'border-blue-200 shadow-sm' : 'border-gray-200 shadow-sm'
            } p-5`}>
              <StageHeader number="1" title="Lexical URL Analysis"
                subtitle="Pattern matching on URL string — no network requests" status={stage1.status} />
              {stage1.status !== 'pending' && (
                <div className="mt-4 ml-11">
                  <div className="flex flex-wrap gap-1.5">
                    {stage1.features.map(([key, value]) => {
                      const risky = isRisky(key, value)
                      return (
                        <span key={key} className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono transition-all ${
                          risky ? 'bg-red-50 text-red-700 border border-red-100' : 'bg-gray-50 text-gray-600 border border-gray-100'
                        }`}>
                          <span className="font-normal text-gray-400 hidden sm:inline">{FEATURE_LABELS[key] || key}:</span>
                          <span className="font-medium">{formatValue(key, value)}</span>
                          {risky && <span className="text-red-400">⚠</span>}
                        </span>
                      )
                    })}
                    {stage1.status === 'running' && (
                      <span className="inline-flex items-center px-2 py-1 rounded-md text-xs text-gray-400 bg-gray-50 border border-gray-100 animate-pulse">scanning...</span>
                    )}
                  </div>
                  {stage1.status === 'done' && stage1.score !== null && <ScoreBar score={stage1.score} />}
                </div>
              )}
            </div>

            {/* Stage 2 */}
            <div className={`bg-white rounded-xl border transition-all duration-300 ${
              stage2.status === 'pending' ? 'border-gray-100 opacity-40' :
              stage2.status === 'running' ? 'border-blue-200 shadow-sm' : 'border-gray-200 shadow-sm'
            } p-5`}>
              <StageHeader number="2" title="ML Classifier — Random Forest"
                subtitle="Trained on phishing URL corpus (Sahingoz et al., 2019 approach)" status={stage2.status} />
              {stage2.status !== 'pending' && (
                <div className="mt-3 ml-11">
                  {stage2.note && <p className={`text-xs font-mono ${stage2.status === 'running' ? 'text-blue-500' : 'text-gray-400'}`}>{stage2.note}</p>}
                  {stage2.status === 'done' && stage2.score !== null && <ScoreBar score={stage2.score} />}
                </div>
              )}
            </div>

            {/* Stage 3 */}
            <div className={`bg-white rounded-xl border transition-all duration-300 ${
              stage3.status === 'pending' ? 'border-gray-100 opacity-40' :
              stage3.status === 'running' ? 'border-blue-200 shadow-sm' : 'border-gray-200 shadow-sm'
            } p-5`}>
              <StageHeader
                number="3"
                title="Visual Brand Matching"
                subtitle={`${modelLabel} embeddings — cosine similarity against 48-brand knowledge base`}
                status={stage3.status}
              />
              {stage3.status !== 'pending' && (
                <div className="mt-3 ml-11">
                  {stage3.note && (
                    <p className={`text-xs font-mono ${stage3.status === 'running' ? 'text-blue-500' : 'text-gray-400'}`}>{stage3.note}</p>
                  )}
                  {stage3.status === 'done' && stage3.screenshotUrl && (
                    <div className="mt-2 space-y-2">
                      <img src={stage3.screenshotUrl} alt="Screenshot"
                        className="rounded-lg border border-gray-200 w-full max-h-40 object-cover object-top" />
                      {stage3.matched ? (
                        <p className="text-sm text-gray-700">
                          Brand detected: <span className="font-semibold text-gray-900">{stage3.display}</span>
                          <span className="ml-2 text-xs text-gray-400 font-mono">similarity {((stage3.similarity || 0) * 100).toFixed(1)}%</span>
                          <span className={`ml-2 text-xs px-1.5 py-0.5 rounded font-medium ${visualModel === 'dino' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}>
                            {modelLabel}
                          </span>
                        </p>
                      ) : (
                        <p className="text-sm text-gray-400">
                          No brand match found
                          {stage3.similarity !== null && (
                            <span className="ml-2 text-xs font-mono">best similarity {((stage3.similarity || 0) * 100).toFixed(1)}%</span>
                          )}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {result && phase === 'done' && (
              <div className="pt-2">
                <VerdictCard
                  verdict={result.verdict}
                  lexicalScore={result.lexical_score}
                  mlScore={result.ml_score}
                  visualBrand={stage3.matched ? stage3.display : null}
                />
              </div>
            )}

          </div>
        )}
      </main>
    </div>
  )
}