// Root application component. Manages a state machine: landing → loading → quiz → results.
// All API orchestration for the upload/ingest flow lives in LandingScreen.
// quizKey increments each time the user returns to the quiz, forcing a clean remount.
//
// Created: 2026-04-02
// Updated: 2026-04-10 — added results screen; per-topic score breakdown; quizKey reset
// Author: Devon Vanaenrode

import './App.css'
import { useState } from 'react'
import LandingScreen from '@/screens/LandingScreen'
import LoadingScreen from '@/screens/LoadingScreen'
import QuizScreen    from '@/screens/QuizScreen'
import ResultsScreen from '@/screens/ResultsScreen'
import type { TopicScores } from '@/screens/QuizScreen'

type Screen = 'landing' | 'loading' | 'quiz' | 'results'

export default function App() {
  const [screen,    setScreen]    = useState<Screen>('landing')
  const [zones,     setZones]     = useState<string[]>([])
  const [error,     setError]     = useState<string | null>(null)
  const [breakdown, setBreakdown] = useState<TopicScores>({})
  // Incrementing this key forces QuizScreen to remount with a clean state
  const [quizKey,   setQuizKey]   = useState(0)

  function handleLoading() {
    setError(null)
    setScreen('loading')
  }

  function handleReady(z: string[]) {
    setZones(z)
    setScreen('quiz')
  }

  function handleError(msg: string) {
    setError(msg)
    setScreen('landing')
  }

  function handleResults(scores: TopicScores) {
    setBreakdown(scores)
    setScreen('results')
  }

  // Return to quiz setup with a fresh state, preserving the current document zones
  function handleQuizAgain() {
    setQuizKey((k) => k + 1)
    setScreen('quiz')
  }

  if (screen === 'loading') {
    return <LoadingScreen />
  }

  if (screen === 'quiz') {
    return (
      <QuizScreen
        key={quizKey}
        zones={zones}
        onResults={handleResults}
        onRestart={() => setScreen('landing')}
      />
    )
  }

  if (screen === 'results') {
    return (
      <ResultsScreen
        breakdown={breakdown}
        onQuizAgain={handleQuizAgain}
        onRestart={() => setScreen('landing')}
      />
    )
  }

  return (
    <LandingScreen
      onLoading={handleLoading}
      onReady={handleReady}
      onError={handleError}
      error={error}
    />
  )
}
