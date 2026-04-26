import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import {
  FiBookOpen,
  FiSearch,
  FiHelpCircle,
  FiFileText,
  FiDatabase,
} from 'react-icons/fi'
import { explainApi } from '../api/explainApi'
import Spinner from '../components/Spinner'

export default function Explain() {
  const [topicSearch, setTopicSearch] = useState('')
  const [activeDisease, setActiveDisease] = useState(null)
  const [diseaseResult, setDiseaseResult] = useState(null)
  const [question, setQuestion] = useState('')
  const [questionResult, setQuestionResult] = useState(null)

  const { data: topics } = useQuery({
    queryKey: ['explain-topics'],
    queryFn: explainApi.listTopics,
  })

  const { data: stats } = useQuery({
    queryKey: ['explain-stats'],
    queryFn: explainApi.getStats,
  })

  const diseaseMutation = useMutation({
    mutationFn: (name) => explainApi.explainDisease(name, 5),
    onSuccess: (data) => {
      setDiseaseResult(data)
    },
    onError: (e) =>
      toast.error(e.response?.data?.detail || 'Could not load explanation'),
  })

  const questionMutation = useMutation({
    mutationFn: (q) => explainApi.ask(q),
    onSuccess: (data) => setQuestionResult(data),
    onError: (e) => toast.error(e.response?.data?.detail || 'Question failed'),
  })

  const diseases = topics?.topics || []
  const filteredDiseases = diseases.filter((d) =>
    d.toLowerCase().includes(topicSearch.toLowerCase())
  )

  const handleSelectDisease = (d) => {
    setActiveDisease(d)
    setDiseaseResult(null)
    diseaseMutation.mutate(d)
  }

  const handleAsk = (e) => {
    e.preventDefault()
    const q = question.trim()
    if (!q) return
    setQuestionResult(null)
    questionMutation.mutate(q)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-ink-900 mb-2 flex items-center gap-2">
          <FiBookOpen /> Knowledge Explorer
        </h1>
        <p className="text-ink-600">
          Evidence-based medical answers powered by RAG. Each response cites its sources.
        </p>
      </div>

      {/* Stats banner */}
      {stats && (
        <div className="card flex flex-wrap gap-6 items-center">
          <div className="flex items-center gap-2">
            <FiDatabase className="text-ink-700" />
            <span className="text-sm text-ink-700">
              <strong>{stats.document_count ?? '—'}</strong> chunks
            </span>
          </div>
          <div className="flex items-center gap-2">
            <FiFileText className="text-ink-700" />
            <span className="text-sm text-ink-700">
              <strong>{stats.topic_count ?? diseases.length}</strong> topics indexed
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Disease browser */}
        <div className="card">
          <h3 className="font-bold text-ink-900 mb-3">Browse diseases</h3>
          <div className="relative mb-3">
            <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
            <input
              value={topicSearch}
              onChange={(e) => setTopicSearch(e.target.value)}
              placeholder="Search diseases..."
              className="input-field pl-10"
            />
          </div>

          <div className="max-h-72 overflow-y-auto pr-1 flex flex-wrap gap-2">
            {filteredDiseases.length === 0 ? (
              <p className="text-sm text-ink-500">No matches.</p>
            ) : (
              filteredDiseases.map((d) => (
                <button
                  key={d}
                  onClick={() => handleSelectDisease(d)}
                  className={`px-3 py-1.5 rounded-full text-sm border transition-colors ${
                    activeDisease === d
                      ? 'bg-ink-900 text-beige-50 border-ink-900'
                      : 'bg-beige-100 text-ink-800 border-beige-300 hover:bg-beige-200'
                  }`}
                >
                  {d}
                </button>
              ))
            )}
          </div>
        </div>

        {/* Free-form question */}
        <div className="card">
          <h3 className="font-bold text-ink-900 mb-3 flex items-center gap-2">
            <FiHelpCircle /> Ask a question
          </h3>
          <form onSubmit={handleAsk} className="space-y-3">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What are the early signs of diabetes?"
              rows={3}
              className="input-field resize-none"
            />
            <button
              type="submit"
              disabled={!question.trim() || questionMutation.isPending}
              className="btn-primary"
            >
              {questionMutation.isPending ? 'Searching...' : 'Ask'}
            </button>
          </form>
        </div>
      </div>

      {/* Disease result */}
      {(diseaseMutation.isPending || diseaseResult) && (
        <div className="card">
          <h3 className="font-bold text-ink-900 mb-3 font-serif text-xl">
            {activeDisease}
          </h3>
          {diseaseMutation.isPending ? (
            <Spinner label="Retrieving evidence..." />
          ) : (
            <ResultBlock result={diseaseResult} />
          )}
        </div>
      )}

      {/* Question result */}
      {(questionMutation.isPending || questionResult) && (
        <div className="card">
          <h3 className="font-bold text-ink-900 mb-3 flex items-center gap-2">
            <FiHelpCircle /> Answer
          </h3>
          {questionMutation.isPending ? (
            <Spinner label="Searching the knowledge base..." />
          ) : (
            <ResultBlock result={questionResult} />
          )}
        </div>
      )}
    </div>
  )
}

function ResultBlock({ result }) {
  if (!result) return null
  const answer = result.explanation || result.answer || result.response
  const sources = result.sources || result.citations || []

  return (
    <div className="space-y-4">
      {answer && (
        <div className="prose prose-sm max-w-none text-ink-800 whitespace-pre-wrap leading-relaxed">
          {answer}
        </div>
      )}

      {sources.length > 0 && (
        <div className="border-t border-beige-200 pt-4">
          <p className="text-xs uppercase tracking-widest text-ink-600 mb-2">
            Sources ({sources.length})
          </p>
          <div className="space-y-2">
            {sources.map((s, i) => (
              <SourceCard key={i} index={i + 1} source={s} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function SourceCard({ index, source }) {
  const title = source.title || source.disease || 'Source'
  const ref = source.source || source.author
  const url = source.url
  const score = source.relevance ?? source.score ?? source.similarity
  const text = source.text || source.content || source.chunk

  return (
    <div className="rounded-lg border border-beige-200 bg-beige-50 p-3 text-sm">
      <div className="flex items-center justify-between mb-1 gap-2">
        <span className="font-semibold text-ink-900">
          [{index}] {title}
        </span>
        {score != null && (
          <span className="text-xs text-ink-600 whitespace-nowrap">
            relevance: {Number(score).toFixed(3)}
          </span>
        )}
      </div>
      {ref && <p className="text-xs text-ink-600 italic mb-1">{ref}</p>}
      {text && <p className="text-ink-700 line-clamp-3">{text}</p>}
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-ink-900 underline hover:text-beige-700"
        >
          View source &rarr;
        </a>
      )}
    </div>
  )
}
