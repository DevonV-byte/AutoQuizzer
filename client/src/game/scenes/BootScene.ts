// BootScene — placeholder scene shown while the real world scenes are built.
// Displays the game title and the names of all zones loaded from the backend.
// Will be replaced by a proper world map scene in a later step.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import Phaser from 'phaser'
import type { Zone } from '@/api/grimoire'
import { GAME_WIDTH, GAME_HEIGHT } from '@/game/config'

export class BootScene extends Phaser.Scene {
  private zones: Zone[]

  constructor(zones: Zone[]) {
    super({ key: 'BootScene' })
    this.zones = zones
  }

  create() {
    const cx = GAME_WIDTH / 2
    const cy = GAME_HEIGHT / 2

    this.add
      .text(cx, cy - 40, 'Grimoire', {
        fontSize: '40px',
        color: '#9b59ff',
        fontFamily: 'monospace',
      })
      .setOrigin(0.5)

    const names = this.zones.map((z) => z.name).join('  |  ') || 'No zones loaded'
    this.add
      .text(cx, cy + 20, names, {
        fontSize: '12px',
        color: '#6b6b88',
        fontFamily: 'monospace',
      })
      .setOrigin(0.5)
  }
}
