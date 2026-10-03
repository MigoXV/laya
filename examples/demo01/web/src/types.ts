export type QuestionType = 'choice' | 'score' | 'noul'

export interface DecisionRequest {
  state: string
  questions: {
    q1: {
      type: QuestionType
      instructions: string
      criteria?: string[]
    }
  }
}

export interface ModelInfo {
  model: string
  device: string
  dtype: string
  runner: string
  torch_version: string
  fingerprint: string
  max_len: number
  calibrated: boolean
  ready: boolean
}

export interface Answer {
  type: QuestionType
  choice?: string
  score?: number
  noul?: number
  confidence: number
  act_probability: number
  probabilities: Record<string, number>
}

export interface DecisionResponse {
  model: ModelInfo
  answers: Record<string, Answer>
  usage: { input_tokens: number; output_tokens: number }
  timings: { preprocess_ms: number; inference_ms: number; total_ms: number; queue_ms: number }
}
