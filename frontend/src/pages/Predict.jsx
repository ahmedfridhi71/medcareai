import { useState, useMemo } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { FiSearch, FiX, FiPlus, FiActivity, FiInfo } from 'react-icons/fi'
import { predictApi } from '../api/predictApi'
import Spinner from '../components/Spinner'

function formatSymptom(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export default function Predict() {
  const [selected, setSelected] = useState([])
  const [search, setSearch] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [explanation, setExplanation] = useState(null)

  const { data: symptomsData, isLoading: loadingSymptoms } = useQuery({
    queryKey: ['symptoms'],
    queryFn: predictApi.listSymptoms,
  })

  const allSymptoms = symptomsData?.symptoms || []

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return allSymptoms.slice(0, 30)
    return allSymptoms
      .filter((s) => s.toLowerCase().includes(q) && !selected.includes(s))
      .slice(0, 30)
  }, [search, allSymptoms, selected])

  const predictMutation = useMutation({
    mutationFn: () => predictApi.predict(selected, 5),
    onSuccess: (data) => {
      setPrediction(data)
      setExplanation(null)
      toast.success('Prediction generated')
    },
    onError: (e) => toast.error(e.response?.data?.detail || 'Prediction failed'),
  })

  const explainMutation = useMutation({
    mutationFn: () => predictApi.explain(selected),
    onSuccess: (data) => {
      setExplanation(data)
      toast.success('Explanation ready')
    },
    onError: (e) => toast.error(e.response?.data?.detail || 'Explanation failed'),
  })

  const addSymptom = (s) => {
    if (!selected.includes(s)) setSelected([...selected, s])
    setSearch('')
  }
  const removeSymptom = (s) => setSelected(selected.filter((x) => x !== s))
  const reset = () => {
    setSelected([])
    setPrediction(null)
    setExplanation(null)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
      {/* Left column */}
      <div className="lg:col-span-3 space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-ink-900 mb-2">Disease Predictor</h1>
          <p className="text-ink-600">
            Pick your symptoms — our model returns the most likely diseases with explainability.
          </p>
        </div>

        {/* Search */}
        <div className="card">
          <label className="block text-sm font-medium text-ink-800 mb-2">
            Search symptoms ({allSymptoms.length} available)
          </label>
          <div className="relative">
            <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="e.g. fever, headache, cough..."
              className="input-field pl-10"
              disabled={loadingSymptoms}
            />
          </div>

          {loadingSymptoms ? (
            <div className="mt-4">
              <Spinner label="Loading symptoms..." />
            </div>
          ) : (
            <div className="mt-4 flex flex-wrap gap-2 max-h-56 overflow-y-auto">
              {filtered.map((s) => (
                <button
                  key={s}
                  onClick={() => addSymptom(s)}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full bg-beige-100 text-ink-800 text-sm hover:bg-ink-900 hover:text-beige-50 transition-colors border border-beige-300"
                >
                  <FiPlus size={12} /> {formatSymptom(s)}
                </button>
              ))}
              {filtered.length === 0 && (
                <p className="text-sm text-ink-500">No matching symptoms.</p>
              )}
            </div>
          )}
        </div>

        {/* Selected */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-ink-900">
              Selected symptoms ({selected.length})
            </h3>
            {selected.length > 0 && (
              <button onClick={reset} className="text-xs text-ink-600 hover:text-ink-900 underline">
                Clear all
              </button>
            )}
          </div>
          {selected.length === 0 ? (
            <p className="text-sm text-ink-500 italic">No symptoms selected yet.</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {selected.map((s) => (
                <span
                  key={s}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-ink-900 text-beige-50 text-sm"
                >
                  {formatSymptom(s)}
                  <button
                    onClick={() => removeSymptom(s)}
                    className="hover:text-beige-300"
                    aria-label="Remove"
                  >
                    <FiX size={14} />
                  </button>
                </span>
              ))}
            </div>
          )}

          <div className="flex flex-wrap gap-3 mt-5">
            <button
              onClick={() => predictMutation.mutate()}
              disabled={selected.length === 0 || predictMutation.isPending}
              className="btn-primary inline-flex items-center gap-2"
            >
              <FiActivity />
              {predictMutation.isPending ? 'Predicting...' : 'Predict Disease'}
            </button>
            <button
              onClick={() => explainMutation.mutate()}
              disabled={selected.length === 0 || explainMutation.isPending}
              className="btn-outline inline-flex items-center gap-2"
            >
              <FiInfo />
              {explainMutation.isPending ? 'Explaining...' : 'Explain (SHAP)'}
            </button>
          </div>
        </div>
      </div>

      {/* Right column */}
      <div className="lg:col-span-2 space-y-6">
        <div className="card">
          <h3 className="font-bold text-ink-900 mb-4 flex items-center gap-2">
            <FiActivity /> Top Predictions
          </h3>
          {!prediction ? (
            <p className="text-sm text-ink-500 italic">
              Run a prediction to see disease probabilities here.
            </p>
          ) : (
            <div className="space-y-3">
              <div className="p-4 rounded-lg bg-ink-900 text-beige-50">
                <div className="text-xs uppercase tracking-widest text-beige-300">
                  Most likely
                </div>
                <div className="text-xl font-serif font-bold capitalize">
                  {prediction.primary_prediction}
                </div>
                <div className="text-sm text-beige-200 mt-1 flex flex-wrap gap-3">
                  <span>Confidence: {(prediction.confidence * 100).toFixed(2)}%</span>
                  {prediction.icd10_code && (
                    <span>ICD-10: {prediction.icd10_code}</span>
                  )}
                  {prediction.severity && (
                    <span className="capitalize">{prediction.severity}</span>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                {prediction.top_predictions?.map((p, i) => (
                  <div key={`${p.disease}-${i}`}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-ink-800 font-medium capitalize">
                        {i + 1}. {p.disease}
                      </span>
                      <span className="text-ink-600">
                        {(p.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="h-2 bg-beige-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-ink-900 rounded-full"
                        style={{ width: `${Math.min(p.confidence * 100 * 20, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {explanation && (
          <div className="card">
            <h3 className="font-bold text-ink-900 mb-3 flex items-center gap-2">
              <FiInfo /> SHAP Explanation
            </h3>
            <p className="text-sm text-ink-700 mb-4">
              Top contributing features for{' '}
              <strong className="capitalize">{explanation.disease}</strong>:
            </p>
            <ContributorList
              title="Positive contributors"
              items={explanation.positive_contributors}
              positive
            />
            <ContributorList
              title="Negative contributors"
              items={explanation.negative_contributors}
            />
            {explanation.base_value != null && (
              <p className="text-xs text-ink-500 mt-3">
                Base value: {explanation.base_value.toFixed(4)}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ContributorList({ title, items, positive }) {
  if (!items || items.length === 0) return null
  const max = Math.max(...items.map((i) => Math.abs(i.contribution)), 0.0001)
  return (
    <div className="mb-3">
      <p className="text-xs uppercase tracking-widest text-ink-600 mb-2">{title}</p>
      <div className="space-y-2">
        {items.map((it) => {
          const pct = Math.min((Math.abs(it.contribution) / max) * 100, 100)
          return (
            <div key={it.symptom}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-ink-800 capitalize">{it.symptom}</span>
                <span className={positive ? 'text-ink-900' : 'text-beige-700'}>
                  {it.contribution >= 0 ? '+' : ''}
                  {it.contribution.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 bg-beige-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${positive ? 'bg-ink-900' : 'bg-beige-500'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
