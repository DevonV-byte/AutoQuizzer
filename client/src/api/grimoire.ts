// Typed fetch client for the Grimoire FastAPI backend.
// All interfaces mirror the Pydantic models in Code/Backend/main.py
// (canonical types live in Code/types/api.ts, generated from the OpenAPI spec).
// Functions map 1-to-1 with the 8 API endpoints.
// Created: 2026-04-02
// Author: Devon Vanaenrode

// ---------------------------------------------------------------------------
// Shared enums
// ---------------------------------------------------------------------------

export type Difficulty = 'easy' | 'medium' | 'hard'
export type EnemyTier = 'Slime' | 'Skeleton' | 'Dragon'

// ---------------------------------------------------------------------------
// Shared sub-shapes
// ---------------------------------------------------------------------------

export interface DifficultyInfo {
  key: Difficulty
  /** Display label, e.g. "Tier 1 – Slime-level" */
  label: string
  /** Bloom's taxonomy guidance for this tier */
  description: string
}

export interface Zone {
  /** Matches the topic_cluster metadata value in ChromaDB */
  name: string
  enemy_tier: EnemyTier
  chunk_count: number
}

export interface QuizQuestion {
  question_number: number
  question: string
  /** Answer options keyed by letter, e.g. { A: "...", B: "..." } */
  options: Record<string, string>
  /** Letter key of the correct answer */
  answer: string
  explanation: string
}

// ---------------------------------------------------------------------------
// Request interfaces
// ---------------------------------------------------------------------------

export interface QuizRequest {
  topic: string
  difficulty?: Difficulty
  n_questions?: number
  n_options?: number
}

export interface EncounterRequest {
  zone: string
  difficulty?: Difficulty
}

export interface AnswerRequest {
  question: string
  correct_answer: string
  player_answer: string
  difficulty: Difficulty
  explanation?: string
}

// ---------------------------------------------------------------------------
// Response interfaces
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: 'ok'
  version: string
}

export interface DifficultiesResponse {
  difficulties: DifficultyInfo[]
}

export interface QuizResponse {
  quiz_title: string
  /** Full tier label, e.g. "Tier 2 – Skeleton-level" */
  difficulty: string
  questions: QuizQuestion[]
}

export interface UploadResponse {
  chunks_added: number
}

export interface GenerateWorldResponse {
  zones: Zone[]
}

export interface ZonesResponse {
  zones: string[]
}

export interface EncounterResponse {
  zone: string
  difficulty: Difficulty
  question: string
  options: Record<string, string>
  answer: string
  explanation: string
}

export interface AnswerResponse {
  correct: boolean
  /** XP gained: 10/20/30 for easy/medium/hard on correct; 0 on wrong */
  xp_delta: number
  /** HP change: 0 on correct; -10/-20/-30 for easy/medium/hard on wrong */
  hp_delta: number
  difficulty: Difficulty
  explanation: string
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Resolves to '' in dev (Vite proxy forwards /api → localhost:8000)
// Set VITE_API_URL in production to the deployed backend origin
const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method }

  if (body !== undefined) {
    if (body instanceof FormData) {
      init.body = body
    } else {
      init.headers = { 'Content-Type': 'application/json' }
      init.body = JSON.stringify(body)
    }
  }

  const res = await fetch(`${BASE}${path}`, init)

  if (!res.ok) {
    // FastAPI returns { detail: string } for 4xx/5xx
    let detail = res.statusText
    try {
      const json = (await res.json()) as { detail?: string }
      if (json.detail) detail = String(json.detail)
    } catch {
      // ignore parse errors on error bodies
    }
    throw new ApiError(res.status, detail)
  }

  return res.json() as Promise<T>
}

const get = <T>(path: string) => request<T>('GET', path)
const post = <T>(path: string, body?: unknown) => request<T>('POST', path, body)

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** GET /health — liveness check */
export const health = () => get<HealthResponse>('/api/health')

/** GET /quiz/difficulties — list all difficulty tiers */
export const getDifficulties = () => get<DifficultiesResponse>('/api/quiz/difficulties')

/** POST /quiz/generate — generate a full quiz via the RAG pipeline */
export const generateQuiz = (req: QuizRequest) => post<QuizResponse>('/api/quiz/generate', req)

/** POST /upload — ingest documents into ChromaDB */
export const uploadDocuments = (files: File[]): Promise<UploadResponse> => {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  return post<UploadResponse>('/api/upload', form)
}

/** POST /generate-world — build game zones from ChromaDB metadata */
export const generateWorld = () => post<GenerateWorldResponse>('/api/generate-world')

/** GET /zones — list available zone names */
export const getZones = () => get<ZonesResponse>('/api/zones')

/** POST /encounter — generate a single combat question for a zone */
export const encounter = (req: EncounterRequest) =>
  post<EncounterResponse>('/api/encounter', req)

/** POST /answer — evaluate player answer, returns XP/HP deltas */
export const submitAnswer = (req: AnswerRequest) => post<AnswerResponse>('/api/answer', req)
