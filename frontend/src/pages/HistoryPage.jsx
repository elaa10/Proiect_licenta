import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'

const VERDICT_CONFIG = {
  legitimate: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', label: 'Legitimate' },
  suspicious:  { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-200',   label: 'Suspicious'  },
  phishing:    { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-200',     label: 'Phishing'    },
}

function VerdictBadge({ verdict }) {
  const c = VERDICT_CONFIG[verdict] || { bg: 'bg-gray-50', text: 'text-gray-500', border: 'border-gray-200', label: verdict || 'Unknown' }
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${c.bg} ${c.text} ${c.border}`}>
      {c.label}
    </span>
  )
}

function formatDate(iso) {
  return new Date(iso).toLocaleString('ro-RO', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export default function HistoryPage() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    api.get('/analyze/history')
      .then(res => setItems(res.data))
      .catch(() => setError('Could not load history.'))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/dashboard')} className="text-sm text-gray-400 hover:text-gray-600">← Dashboard</button>
          <span className="text-gray-300">|</span>
          <h1 className="text-lg font-bold text-gray-800">Analysis History</h1>
        </div>
        <button
          onClick={() => navigate('/analyze')}
          className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
        >
          New analysis
        </button>
      </nav>

      <main className="max-w-3xl mx-auto px-4 py-10">
        {loading && (
          <div className="text-center text-gray-400 py-20 text-sm">Loading...</div>
        )}
        {error && (
          <div className="text-center text-red-500 py-20 text-sm">{error}</div>
        )}
        {!loading && !error && items.length === 0 && (
          <div className="text-center py-20">
            <p className="text-gray-400 text-sm">No analyses yet.</p>
            <button
              onClick={() => navigate('/analyze')}
              className="mt-4 px-5 py-2.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
            >
              Analyze your first URL
            </button>
          </div>
        )}
        {!loading && items.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs text-gray-400 mb-4">{items.length} analysis{items.length !== 1 ? 'es' : ''} found</p>
            {items.map(item => (
              <div
                key={item.request_id}
                className="bg-white rounded-xl border border-gray-200 px-5 py-4 flex items-center gap-4 hover:shadow-sm transition-shadow"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-mono text-gray-800 truncate">{item.url}</p>
                  <p className="text-xs text-gray-400 mt-1">{formatDate(item.created_at)}</p>
                </div>
                <VerdictBadge verdict={item.verdict} />
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}