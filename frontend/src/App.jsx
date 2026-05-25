import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import AnalyzePage from './pages/AnalyzePage'
import ProtectedRoute from './components/ProtectedRoute'
import HistoryPage from './pages/HistoryPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/analyze" element={<ProtectedRoute><AnalyzePage /></ProtectedRoute>} />
        <Route path="*" element={<Navigate to="/login" replace />} />
        <Route path="/history" element={<ProtectedRoute><HistoryPage /></ProtectedRoute>} />
      </Routes>
    </BrowserRouter>
  )
}