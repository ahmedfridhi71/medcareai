import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Predict from './pages/Predict'
import Chat from './pages/Chat'
import Explain from './pages/Explain'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="predict" element={<Predict />} />
        <Route path="chat" element={<Chat />} />
        <Route path="explain" element={<Explain />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}

function NotFound() {
  return (
    <div className="text-center py-24">
      <h1 className="text-6xl font-serif font-bold text-ink-900">404</h1>
      <p className="text-ink-600 mt-2">Page not found.</p>
    </div>
  )
}

export default App
