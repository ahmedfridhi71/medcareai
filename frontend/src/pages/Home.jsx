import { Link } from 'react-router-dom'
import {
  FiActivity,
  FiMessageSquare,
  FiBookOpen,
  FiArrowRight,
  FiShield,
  FiCpu,
  FiDatabase,
} from 'react-icons/fi'

const features = [
  {
    icon: FiActivity,
    title: 'Symptom-Based Prediction',
    desc: 'Select your symptoms and get a top-K disease prediction with SHAP explanations.',
    to: '/predict',
    cta: 'Try Predictor',
  },
  {
    icon: FiMessageSquare,
    title: 'Conversational Triage',
    desc: 'Chat with our Mistral-powered medical assistant to describe your symptoms naturally.',
    to: '/chat',
    cta: 'Start Chat',
  },
  {
    icon: FiBookOpen,
    title: 'Evidence-Based Explanations',
    desc: 'Ask any medical question — answers are backed by RAG-retrieved medical sources.',
    to: '/explain',
    cta: 'Explore Knowledge',
  },
]

const stats = [
  { icon: FiCpu, value: '377', label: 'Symptoms Tracked' },
  { icon: FiDatabase, value: 'RAG', label: 'Source-Cited Answers' },
  { icon: FiShield, value: 'SHAP', label: 'Explainable AI' },
]

export default function Home() {
  return (
    <div className="space-y-16">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl bg-ink-900 text-beige-50 px-6 sm:px-12 py-16 sm:py-24">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute -top-12 -right-12 w-72 h-72 rounded-full bg-beige-300 blur-3xl" />
          <div className="absolute bottom-0 left-0 w-96 h-96 rounded-full bg-beige-500 blur-3xl" />
        </div>

        <div className="relative max-w-3xl">
          <span className="inline-block badge bg-beige-200 text-ink-900 mb-6">
            ML &middot; LLM &middot; RAG
          </span>
          <h1 className="font-serif text-4xl sm:text-6xl font-bold leading-tight mb-6">
            Smarter medical decisions, <span className="text-beige-300">explained</span>.
          </h1>
          <p className="text-lg text-beige-200 mb-8 max-w-2xl">
            MedCareAI combines machine learning, large language models, and retrieval-augmented
            generation to deliver transparent, evidence-based clinical decision support.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link
              to="/predict"
              className="inline-flex items-center gap-2 bg-beige-200 text-ink-900 px-6 py-3 rounded-lg font-medium hover:bg-beige-300 transition-colors"
            >
              Predict a Disease <FiArrowRight />
            </Link>
            <Link
              to="/chat"
              className="inline-flex items-center gap-2 border-2 border-beige-300 text-beige-100 px-6 py-3 rounded-lg font-medium hover:bg-beige-300 hover:text-ink-900 transition-colors"
            >
              Talk to the Assistant
            </Link>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {stats.map(({ icon: Icon, value, label }) => (
          <div
            key={label}
            className="card flex items-center gap-4 hover:shadow-warm transition-shadow"
          >
            <div className="w-12 h-12 rounded-lg bg-ink-900 text-beige-100 flex items-center justify-center">
              <Icon size={22} />
            </div>
            <div>
              <div className="text-2xl font-serif font-bold text-ink-900">{value}</div>
              <div className="text-sm text-ink-600">{label}</div>
            </div>
          </div>
        ))}
      </section>

      {/* Features */}
      <section>
        <div className="text-center mb-10">
          <h2 className="text-3xl sm:text-4xl font-bold text-ink-900 mb-2">What you can do</h2>
          <p className="text-ink-600">Three integrated tools, one unified platform.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map(({ icon: Icon, title, desc, to, cta }) => (
            <Link
              key={to}
              to={to}
              className="card group hover:shadow-warm hover:border-ink-900 transition-all flex flex-col"
            >
              <div className="w-12 h-12 rounded-lg bg-beige-200 group-hover:bg-ink-900 group-hover:text-beige-100 text-ink-900 flex items-center justify-center mb-4 transition-colors">
                <Icon size={22} />
              </div>
              <h3 className="text-xl font-bold text-ink-900 mb-2">{title}</h3>
              <p className="text-ink-600 mb-4 flex-1">{desc}</p>
              <span className="inline-flex items-center gap-1 text-ink-900 font-medium group-hover:gap-2 transition-all">
                {cta} <FiArrowRight />
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
