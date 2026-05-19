import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import api from '../services/api'

export default function Dashboard() {
  const [user, setUser] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    api.get('/auth/me')
      .then(res => setUser(res.data))
      .catch(() => navigate('/login'))
  }, [])

  function logout() {
    localStorage.removeItem('token')
    navigate('/login')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm px-6 py-4 flex justify-between items-center">
        <h1 className="text-xl font-bold text-gray-800">Phishing Detector</h1>
        <div className="flex items-center gap-4">
          {user && <span className="text-sm text-gray-500">{user.email}</span>}
          <button onClick={logout} className="text-sm text-red-500 hover:underline">
            Logout
          </button>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto mt-16 px-4">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome back</h2>
        <p className="text-gray-500 mb-8">
          Use the phishing detector to analyze any URL using lexical analysis, machine learning, and visual brand matching.
        </p>

        <Link
          to="/analyze"
          className="block bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-blue-200 transition-all group"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-800 group-hover:text-blue-600 transition-colors">
                Analyze a URL →
              </h3>
              <p className="text-sm text-gray-500 mt-1">
                Three-stage detection: lexical · ML classifier · visual matching
              </p>
            </div>
            <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600 text-xl group-hover:bg-blue-100 transition-colors">
              🔍
            </div>
          </div>
        </Link>
      </main>
    </div>
  )
}