/**
 * Explain API - RAG-based medical knowledge with source citations.
 */
import axiosClient from './axiosClient'

export const explainApi = {
  /** List available diseases/topics in the knowledge base. */
  listTopics: async () => {
    const { data } = await axiosClient.get('/explain/')
    return data
  },

  /** Get knowledge-base statistics. */
  getStats: async () => {
    const { data } = await axiosClient.get('/explain/stats')
    return data
  },

  /** Get an evidence-based explanation for a disease (RAG + Mistral). */
  explainDisease: async (disease_name, n_chunks = 5) => {
    const { data } = await axiosClient.get(
      `/explain/${encodeURIComponent(disease_name)}`,
      { params: { n_chunks } }
    )
    return data
  },

  /** Ask a free-form medical question. */
  ask: async (question) => {
    const { data } = await axiosClient.post('/explain/question', { question })
    return data
  },
}
