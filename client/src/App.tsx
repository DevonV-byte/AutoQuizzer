// Root application component. Manages a simple state machine that transitions
// between the three top-level screens: landing → loading → game.
// All API orchestration for the startup flow lives here.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import './App.css'
import { useState } from 'react'
import type { Zone } from '@/api/grimoire'
import LandingScreen from '@/screens/LandingScreen'
import LoadingScreen from '@/screens/LoadingScreen'
import GameScreen from '@/screens/GameScreen'

type Screen = 'landing' | 'loading' | 'game'

export default function App() {
  const [screen, setScreen] = useState<Screen>('landing')
  const [zones, setZones] = useState<Zone[]>([])
  const [error, setError] = useState<string | null>(null)

  function handleLoading() {
    setError(null)
    setScreen('loading')
  }

  function handleReady(z: Zone[]) {
    setZones(z)
    setScreen('game')
  }

  function handleError(msg: string) {
    setError(msg)
    setScreen('landing')
  }

  if (screen === 'loading') return <LoadingScreen />
  if (screen === 'game') return <GameScreen zones={zones} />

  return (
    <LandingScreen
      onLoading={handleLoading}
      onReady={handleReady}
      onError={handleError}
      error={error}
    />
  )
}
