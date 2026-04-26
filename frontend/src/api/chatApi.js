/**
 * Chat API - Conversational symptom-collection agent (Mistral).
 */
import axiosClient from './axiosClient'

export const chatApi = {
  /** Start a new chat session. Returns { session_id, message }. */
  start: async () => {
    const { data } = await axiosClient.post('/chat/start', {})
    return data
  },

  /** Send a user message. Returns { reply, ready_for_prediction, ... }. */
  sendMessage: async (session_id, message) => {
    const { data } = await axiosClient.post('/chat/message', { session_id, message })
    return data
  },

  /** Finalize the chat -> get the ML disease prediction. */
  finalize: async (session_id) => {
    const { data } = await axiosClient.post('/chat/finalize', { session_id })
    return data
  },

  /** Get chat history. */
  getHistory: async (session_id) => {
    const { data } = await axiosClient.get(`/chat/history/${session_id}`)
    return data
  },

  /** Delete a chat session. */
  deleteSession: async (session_id) => {
    const { data } = await axiosClient.delete(`/chat/${session_id}`)
    return data
  },

  /** Get total active session count. */
  getCount: async () => {
    const { data } = await axiosClient.get('/chat/sessions/count')
    return data
  },
}
