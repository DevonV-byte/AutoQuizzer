// Typed fetch client for the Grimoire AutoQuizzer backend.
// All interfaces mirror the Pydantic models in Code/Backend/main.py.
// Functions map 1-to-1 with the active API endpoints.
//
// Created: 2026-04-02
// Updated: 2026-04-20 — added EncounterBatchRequest/Response, PlayerProgress types;
//                        updated AnswerRequest/Response; added FileEntry/FilesResponse
//                        and getFiles/deleteFile/clearAllFiles functions
// Author: Devon Vanaenrode

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export type Difficulty = 'easy' | 'medium' | 'hard'

export interface DifficultyInfo {
  key: Difficulty
  /** Display label, e.g. "Tier 1 – Slime-level" */
  label: string
  /** Bloom's taxonomy guidance for this tier */
  description: string
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

export interface EncounterBatchRequest {
  player_id: string
  zone: string
  difficulty?: Difficulty
}

export interface AnswerRequest {
  player_id: string
  zone: string
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

export interface EncounterBatchResponse {
  zone: string
  difficulty: Difficulty
  /** Which quiz batch this is: 1 = first 5 questions, 2 = second 5, etc. */
  quiz_number: number
  questions: EncounterResponse[]
}

export interface AnswerResponse {
  correct: boolean
  /** XP gained: 10/20/30 for easy/medium/hard on correct; 0 on wrong */
  xp_delta: number
  /** HP change: 0 on correct; -10/-20/-30 for easy/medium/hard on wrong */
  hp_delta: number
  difficulty: Difficulty
  explanation: string
  /** True when the 5th answer of a quiz batch is recorded */
  quiz_complete: boolean
  /** Correct answers in the completed quiz (0–5); null if not yet complete */
  quiz_score: number | null
  /** True when all 5 quiz batches in the level are passed at ≥ 4/5 */
  level_complete: boolean
}

export interface PlayerProgressEntry {
  zone: string
  difficulty: Difficulty
  questions_answered: number
  correct_count: number
  quizzes_passed: number
  level_complete: boolean
}

export interface PlayerProgressResponse {
  player_id: string
  progress: PlayerProgressEntry[]
}

export interface FileEntry {
  filename: string
  /** ISO 8601 UTC timestamp */
  uploaded_at: string
}

export interface FilesResponse {
  files: FileEntry[]
}

export interface MessageResponse {
  message: string
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

const get  = <T>(path: string)               => request<T>('GET',  path)
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

/** GET /zones — list topic names derived from uploaded documents */
export const getZones = () => get<ZonesResponse>('/api/zones')

/** POST /encounter — generate a single quiz question for a topic (legacy) */
export const encounter = (req: EncounterRequest) =>
  post<EncounterResponse>('/api/encounter', req)

/** POST /encounter/batch — fetch next 5 questions from the pre-generated pool */
export const encounterBatch = (req: EncounterBatchRequest) =>
  post<EncounterBatchResponse>('/api/encounter/batch', req)

/** POST /answer — evaluate player answer, returns correctness, score deltas, and progress */
export const submitAnswer = (req: AnswerRequest) => post<AnswerResponse>('/api/answer', req)

/** GET /player/{player_id}/progress — per-zone/difficulty progress for a player */
export const getPlayerProgress = (playerId: string) =>
  get<PlayerProgressResponse>(`/api/player/${playerId}/progress`)

/** GET /files — list all files currently in the database */
export const getFiles = () => get<FilesResponse>('/api/files')

/** DELETE /files/{filename} — remove a specific file and its ChromaDB chunks */
export const deleteFile = (filename: string) =>
  request<MessageResponse>('DELETE', `/api/files/${encodeURIComponent(filename)}`)

/** DELETE /files — wipe the entire database (ChromaDB + all progress) */
export const clearAllFiles = () => request<MessageResponse>('DELETE', '/api/files')
