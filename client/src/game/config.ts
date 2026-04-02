// Phaser game configuration factory.
// Accepts the list of scenes so callers can swap scene sets without touching
// renderer or physics settings. The parent element ID must match the div
// rendered by GameScreen.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import Phaser from 'phaser'

export const GAME_WIDTH = 800
export const GAME_HEIGHT = 600
export const PARENT_ID = 'phaser-container'

// Returns a fully-typed GameConfig ready to pass to `new Phaser.Game(config)`.
// Scenes are injected so this file stays scene-agnostic.
export function makeGameConfig(
  scenes: Phaser.Types.Core.GameConfig['scene']
): Phaser.Types.Core.GameConfig {
  return {
    type: Phaser.AUTO,
    width: GAME_WIDTH,
    height: GAME_HEIGHT,
    backgroundColor: '#000000',
    parent: PARENT_ID,
    scene: scenes,
    physics: {
      default: 'arcade',
      arcade: { gravity: { x: 0, y: 0 }, debug: false },
    },
    scale: {
      mode: Phaser.Scale.FIT,
      autoCenter: Phaser.Scale.CENTER_BOTH,
    },
  }
}
