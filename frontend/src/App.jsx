import { Routes, Route } from 'react-router-dom'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </div>
  )
}

function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8">
      <div className="card max-w-lg text-center">
        <h1 className="text-4xl font-bold text-primary-600 mb-4">
          MedCareAI
        </h1>
        <p className="text-gray-600 mb-6">
          Advanced Medical Decision Support Platform
        </p>
        <div className="flex gap-4 justify-center">
          <button className="btn-primary">
            Get Started
          </button>
          <button className="btn-secondary">
            Learn More
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
