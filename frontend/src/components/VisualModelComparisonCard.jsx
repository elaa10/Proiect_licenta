import { useEffect, useState } from 'react'
import api from '../services/api'

function escapeCsv(value) {
  return `"${String(value).replace(/"/g, '""')}"`
}

function exportCsv(rows) {
  const header = ['url', 'clip_brand', 'clip_similarity', 'dino_brand', 'dino_similarity', 'agreement']
  const lines = [header.join(',')]
  for (const r of rows) {
    lines.push([
      escapeCsv(r.url),
      escapeCsv(r.clip_brand || 'none'),
      r.clip_similarity ?? '',
      escapeCsv(r.dino_brand || 'none'),
      r.dino_similarity ?? '',
      r.agreement,
    ].join(','))
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'visual_model_comparison.csv'
  a.click()
  URL.revokeObjectURL(url)
}

function SimilarityBar({ label, value, color }) {
  const pct = parseFloat(((value || 0) * 100).toFixed(1))
  
  return (
    <div className="flex-1">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>{label}</span>
        <span className="font-mono">{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

export default function VisualModelComparisonCard() {
  const [items, setItems] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.get('/analyze/visual/comparisons')
      .then(res => setItems(res.data))
      .catch(() => setError('Could not load model comparison data.'))
  }, [])

  if (error) return <p className="text-sm text-red-500">{error}</p>
  if (items === null) return <p className="text-sm text-gray-400">Loading model comparison...</p>

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-gray-800">CLIP vs DINOv2 — same-URL comparison</h3>
        {items.length > 0 && (
          <button
            onClick={() => exportCsv(items)}
            className="text-xs px-3 py-1 border border-gray-200 rounded-lg text-gray-600 hover:bg-gray-50 transition-colors"
          >
            Export CSV
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <p className="text-sm text-gray-400 mt-2">
          Analyze the same URL with both CLIP and DINOv2 (use the visual model selector on the
          analysis page) to see a side-by-side comparison here.
        </p>
      ) : (
        <div className="space-y-4 mt-3 max-h-80 overflow-y-auto pr-1">
          {items.map((item, idx) => (
            <div key={idx} className="border border-gray-100 rounded-lg p-3">
              <p className="text-xs font-mono text-gray-700 truncate mb-2">{item.url}</p>
              <div className="flex items-center gap-4">
                <SimilarityBar label={`CLIP — ${item.clip_brand || 'no match'}`} value={item.clip_similarity} color="#2563eb" />
                <SimilarityBar label={`DINOv2 — ${item.dino_brand || 'no match'}`} value={item.dino_similarity} color="#9333ea" />
              </div>
              <p className={`text-xs mt-2 font-medium ${item.agreement ? 'text-emerald-600' : 'text-amber-600'}`}>
                {item.agreement ? '✓ Same brand detected' : '⚠ Different result per model'}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}