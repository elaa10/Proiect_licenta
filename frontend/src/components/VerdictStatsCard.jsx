import { useEffect, useState } from 'react'
import api from '../services/api'

const VERDICT_COLORS = {
  legitimate: '#10b981',
  suspicious: '#f59e0b',
  phishing: '#ef4444',
  unknown: '#9ca3af',
}

const VERDICT_LABELS = {
  legitimate: 'Legitimate',
  suspicious: 'Suspicious',
  phishing: 'Phishing',
  unknown: 'Unknown',
}

const RADIUS = 60
const CIRCUMFERENCE = 2 * Math.PI * RADIUS
const ORDER = ['legitimate', 'suspicious', 'phishing', 'unknown']

function buildSegments(stats) {
  if (!stats.total) return []
  let offset = 0
  const segments = []
  for (const key of ORDER) {
    const value = stats[key] || 0
    if (value === 0) continue
    const fraction = value / stats.total
    const length = fraction * CIRCUMFERENCE
    segments.push({
      key,
      length,
      offset,
      value,
      percent: Math.round(fraction * 100),
      color: VERDICT_COLORS[key],
    })
    offset += length
  }
  return segments
}

export default function VerdictStatsCard() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.get('/analyze/stats')
      .then(res => setStats(res.data))
      .catch(() => setError('Could not load statistics.'))
  }, [])

  if (error) return <p className="text-sm text-red-500">{error}</p>
  if (!stats) return <p className="text-sm text-gray-400">Loading statistics...</p>

  if (stats.total === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-sm font-semibold text-gray-800 mb-1">Your analysis statistics</h3>
        <p className="text-sm text-gray-400">No analyses yet. Run your first scan to see statistics here.</p>
      </div>
    )
  }

  const segments = buildSegments(stats)

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 className="text-sm font-semibold text-gray-800 mb-4">Your analysis statistics</h3>
      <div className="flex items-center gap-6">
        <svg width="160" height="160" viewBox="0 0 160 160" className="flex-shrink-0">
          <g transform="translate(80, 80) rotate(-90)">
            <circle r={RADIUS} fill="none" stroke="#f3f4f6" strokeWidth="20" />
            {segments.map(seg => (
              <circle
                key={seg.key}
                r={RADIUS}
                fill="none"
                stroke={seg.color}
                strokeWidth="20"
                strokeDasharray={`${seg.length} ${CIRCUMFERENCE - seg.length}`}
                strokeDashoffset={-seg.offset}
              />
            ))}
          </g>
          <text x="80" y="76" textAnchor="middle" className="fill-gray-800" style={{ fontSize: '22px', fontWeight: 700 }}>
            {stats.total}
          </text>
          <text x="80" y="96" textAnchor="middle" className="fill-gray-400" style={{ fontSize: '11px' }}>
            analyses
          </text>
        </svg>

        <div className="flex-1 space-y-2">
          {segments.map(seg => (
            <div key={seg.key} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: seg.color }} />
                <span className="text-gray-600">{VERDICT_LABELS[seg.key]}</span>
              </div>
              <span className="font-medium text-gray-800">{seg.value} ({seg.percent}%)</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}