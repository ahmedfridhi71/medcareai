/**
 * Predict API - Disease prediction endpoints.
 */
import axiosClient from './axiosClient'

export const predictApi = {
  /** Predict disease from a list of symptoms. */
  predict: async (symptoms, top_k = 5) => {
    const { data } = await axiosClient.post('/predict/', { symptoms, top_k })
    return data
  },

  /** Get SHAP-based explanation for a prediction. */
  explain: async (symptoms) => {
    const { data } = await axiosClient.post('/predict/explain', { symptoms })
    return data
  },

  /** List all available symptoms (377 total). */
  listSymptoms: async () => {
    const { data } = await axiosClient.get('/predict/symptoms')
    return data
  },
}
