// Game viewport: mounts the Phaser 3 game instance and destroys it cleanly
// on unmount to avoid duplicate canvases during React HMR.
// The Phaser config is centralised in src/game/config.ts.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import { useEffect, useRef } from 'react'
import Phaser from 'phaser'
import type { Zone } from '@/api/grimoire'
import { makeGameConfig, PARENT_ID } from '@/game/config'
import { BootScene } from '@/game/scenes/BootScene'

interface Props {
  zones: Zone[]
}

export default function GameScreen({ zones }: Props) {
  const gameRef = useRef<Phaser.Game | null>(null)

  useEffect(() => {
    // zones are stable at mount time — game owns them for its lifetime
    gameRef.current = new Phaser.Game(makeGameConfig([new BootScene(zones)]))

    return () => {
      gameRef.current?.destroy(true)
      gameRef.current = null
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="game">
      <div id={PARENT_ID} />
    </div>
  )
}
