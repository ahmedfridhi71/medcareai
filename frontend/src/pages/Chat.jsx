import { useState, useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { FiSend, FiRefreshCw, FiCheckCircle, FiUser, FiCpu } from 'react-icons/fi'
import { chatApi } from '../api/chatApi'
import { predictApi } from '../api/predictApi'
import Spinner from '../components/Spinner'

export default function Chat() {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [readyForPrediction, setReadyForPrediction] = useState(false)
  const [finalResult, setFinalResult] = useState(null)
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, finalResult])

  const startMutation = useMutation({
    mutationFn: chatApi.start,
    onSuccess: (data) => {
      setSessionId(data.session_id)
      setMessages([{ role: 'assistant', content: data.message }])
      setReadyForPrediction(false)
      setFinalResult(null)
    },
    onError: () => toast.error('Could not start a chat session'),
  })

  const sendMutation = useMutation({
    mutationFn: (msg) => chatApi.sendMessage(sessionId, msg),
    onSuccess: (data) => {
      const cleaned = cleanAssistantText(data.response)
      setMessages((m) => [...m, { role: 'assistant', content: cleaned }])
      setReadyForPrediction(!!data.symptoms_ready)
      if (data.symptoms_ready) {
        toast.success('Ready for diagnosis. Click "Get Diagnosis" to continue.')
      }
    },
    onError: () => toast.error('Message failed'),
  })

  const finalizeMutation = useMutation({
    mutationFn: async () => {
      const finalized = await chatApi.finalize(sessionId)
      const symptoms = finalized.symptoms || []
      if (symptoms.length === 0) {
        throw new Error('No symptoms could be extracted from the conversation.')
      }
      const prediction = await predictApi.predict(symptoms, 5)
      return { ...finalized, prediction }
    },
    onSuccess: (data) => {
      setFinalResult(data)
      toast.success('Analysis complete')
    },
    onError: (e) =>
      toast.error(e.response?.data?.detail || e.message || 'Could not finalize'),
  })

  // Auto-start on first mount
  useEffect(() => {
    if (!sessionId && !startMutation.isPending) {
      startMutation.mutate()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleSend = (e) => {
    e?.preventDefault()
    const text = input.trim()
    if (!text || !sessionId || sendMutation.isPending) return
    setMessages((m) => [...m, { role: 'user', content: text }])
    setInput('')
    sendMutation.mutate(text)
  }

  const handleNewChat = () => {
    setSessionId(null)
    setMessages([])
    setReadyForPrediction(false)
    setFinalResult(null)
    startMutation.mutate()
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-[calc(100vh-12rem)]">
      {/* Chat panel */}
      <div className="lg:col-span-3 flex flex-col card p-0 overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-beige-200 bg-beige-100">
          <div>
            <h2 className="font-bold text-ink-900">Medical Assistant</h2>
            <p className="text-xs text-ink-600">
              {sessionId ? `Session: ${sessionId.slice(0, 8)}...` : 'Initializing...'}
            </p>
          </div>
          <button
            onClick={handleNewChat}
            className="text-sm inline-flex items-center gap-1 text-ink-700 hover:text-ink-900"
          >
            <FiRefreshCw /> New chat
          </button>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 space-y-4 bg-beige-50">
          {messages.map((m, i) => (
            <ChatBubble key={i} role={m.role} content={m.content} />
          ))}

          {sendMutation.isPending && (
            <ChatBubble role="assistant" content={<Spinner label="Thinking..." />} />
          )}

          {!finalResult && messages.filter((m) => m.role === 'user').length >= 1 && (
            <div className="rounded-xl bg-ink-900 text-beige-50 p-4">
              <p className="text-sm mb-3">
                {readyForPrediction
                  ? 'I have enough information to analyze your symptoms.'
                  : 'You can finalize at any time to get a diagnosis based on what you have shared.'}
              </p>
              <button
                onClick={() => finalizeMutation.mutate()}
                disabled={finalizeMutation.isPending}
                className="inline-flex items-center gap-2 bg-beige-200 text-ink-900 px-4 py-2 rounded-lg font-medium hover:bg-beige-300 transition-colors disabled:opacity-50"
              >
                <FiCheckCircle />
                {finalizeMutation.isPending ? 'Analyzing...' : 'Get Diagnosis'}
              </button>
            </div>
          )}
        </div>

        <form
          onSubmit={handleSend}
          className="border-t border-beige-200 p-3 bg-white flex gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your symptoms..."
            className="input-field"
            disabled={!sessionId || sendMutation.isPending}
          />
          <button
            type="submit"
            disabled={!input.trim() || !sessionId || sendMutation.isPending}
            className="btn-primary inline-flex items-center gap-2"
          >
            <FiSend />
          </button>
        </form>
      </div>

      {/* Result panel */}
      <div className="lg:col-span-2 space-y-4 overflow-y-auto">
        <div className="card">
          <h3 className="font-bold text-ink-900 mb-3">How it works</h3>
          <ol className="text-sm text-ink-700 space-y-2 list-decimal list-inside">
            <li>Describe how you feel in plain language.</li>
            <li>The assistant asks follow-up questions to gather symptoms.</li>
            <li>When ready, click <em>Get Diagnosis</em> for an ML-based prediction.</li>
          </ol>
        </div>

        {finalResult && (
          <div className="card border-2 border-ink-900">
            <h3 className="font-bold text-ink-900 mb-2 flex items-center gap-2">
              <FiCheckCircle /> Diagnosis
            </h3>
            <div className="p-4 rounded-lg bg-ink-900 text-beige-50 mb-4">
              <div className="text-xs uppercase tracking-widest text-beige-300">
                Most likely
              </div>
              <div className="text-xl font-serif font-bold capitalize">
                {finalResult.prediction?.primary_prediction}
              </div>
              {finalResult.prediction?.confidence != null && (
                <div className="text-sm text-beige-200 mt-1 flex flex-wrap gap-3">
                  <span>
                    Confidence:{' '}
                    {(finalResult.prediction.confidence * 100).toFixed(2)}%
                  </span>
                  {finalResult.prediction.icd10_code && (
                    <span>ICD-10: {finalResult.prediction.icd10_code}</span>
                  )}
                </div>
              )}
            </div>

            {finalResult.prediction?.top_predictions?.map((p, i) => (
              <div key={`${p.disease}-${i}`} className="mb-2">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-ink-800 capitalize">
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

            {finalResult.symptoms && finalResult.symptoms.length > 0 && (
              <div className="mt-4 pt-4 border-t border-beige-200">
                <p className="text-xs text-ink-600 mb-2">Mapped symptoms:</p>
                <div className="flex flex-wrap gap-1">
                  {finalResult.symptoms.map((s) => (
                    <span key={s} className="badge-light capitalize">
                      {s.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ChatBubble({ role, content }) {
  const isUser = role === 'user'
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-beige-300 text-ink-900' : 'bg-ink-900 text-beige-100'
        }`}
      >
        {isUser ? <FiUser size={14} /> : <FiCpu size={14} />}
      </div>
      <div
        className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm whitespace-pre-wrap ${
          isUser
            ? 'bg-ink-900 text-beige-50 rounded-tr-none'
            : 'bg-white border border-beige-200 text-ink-900 rounded-tl-none'
        }`}
      >
        {content}
      </div>
    </div>
  )
}

// Strip the [SYMPTOMS_READY] marker and any trailing ```json ... ``` block
// the LLM emits when it has gathered enough information. Users should only
// see the natural-language portion of the assistant's reply.
function cleanAssistantText(raw) {
  if (!raw || typeof raw !== 'string') return raw
  let text = raw

  // Remove fenced code blocks (```json ... ``` or ``` ... ```)
  text = text.replace(/```[a-zA-Z]*\n?[\s\S]*?```/g, '')

  // Remove the readiness marker (any case, with or without brackets context)
  text = text.replace(/\[SYMPTOMS_READY\]/gi, '')

  // Remove a trailing standalone JSON object the model may emit unfenced
  text = text.replace(/\{\s*"symptoms"\s*:[\s\S]*?\}\s*$/i, '')

  // Collapse extra blank lines
  text = text.replace(/\n{3,}/g, '\n\n').trim()

  // If we stripped everything, show a friendly fallback
  if (!text) {
    text = 'I have enough information now. Click "Get Diagnosis" to see the results.'
  }
  return text
}
