import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'

const CONFIRM_TEXT = 'DELETE'

export default function DeleteAccountCard() {
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [confirmInput, setConfirmInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function handleDelete() {
    setError('')
    setLoading(true)
    try {
      await api.delete('/auth/me')
      localStorage.removeItem('token')
      navigate('/login')
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not delete account.')
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-red-100 p-6">
      <h3 className="text-sm font-semibold text-red-600 mb-1">Danger zone</h3>
      <p className="text-sm text-gray-500 mb-4">
        Permanently delete your account and all associated analysis history. This action cannot be undone.
      </p>

      {!confirmOpen ? (
        <button
          onClick={() => setConfirmOpen(true)}
          className="px-4 py-2 border border-red-200 text-red-600 text-sm rounded-lg hover:bg-red-50 transition-colors"
        >
          Delete account
        </button>
      ) : (
        <div className="space-y-3">
          <p className="text-sm text-gray-600">
            Type <span className="font-mono font-semibold">{CONFIRM_TEXT}</span> to confirm.
          </p>
          <input
            type="text"
            value={confirmInput}
            onChange={e => setConfirmInput(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-200"
            placeholder={CONFIRM_TEXT}
          />
          {error && <p className="text-sm text-red-500">{error}</p>}
          <div className="flex gap-2">
            <button
              onClick={handleDelete}
              disabled={confirmInput !== CONFIRM_TEXT || loading}
              className="px-4 py-2 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
            >
              {loading ? 'Deleting...' : 'Permanently delete'}
            </button>
            <button
              onClick={() => { setConfirmOpen(false); setConfirmInput('') }}
              className="px-4 py-2 border border-gray-200 text-gray-600 text-sm rounded-lg hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}