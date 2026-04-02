// Landing screen: lets the player upload study documents and start the game.
// On submit, calls uploadDocuments then generateWorld via the API client.
// Reports progress to App via onLoading / onReady / onError callbacks.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import { useState } from 'react'
import { uploadDocuments, generateWorld, ApiError } from '@/api/grimoire'
import type { Zone } from '@/api/grimoire'

interface Props {
  onLoading: () => void
  onReady: (zones: Zone[]) => void
  onError: (msg: string) => void
  error: string | null
}

// File types accepted by the backend ingestion pipeline
const ACCEPTED = '.txt,.docx,.ipynb,.py'

export default function LandingScreen({ onLoading, onReady, onError, error }: Props) {
  const [files, setFiles] = useState<File[]>([])

  function handleFiles(e: React.ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(e.target.files ?? []))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (files.length === 0) return

    onLoading()

    try {
      await uploadDocuments(files)
      const world = await generateWorld()
      onReady(world.zones)
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Something went wrong.'
      onError(msg)
    }
  }

  return (
    <div className="screen">
      <h1 className="landing__title">Grimoire</h1>
      <p className="landing__subtitle">Upload your notes. Enter the dungeon.</p>

      <form className="landing__form" onSubmit={handleSubmit}>
        <label className="landing__file-label">
          {files.length === 0 ? 'Click to choose files' : `${files.length} file(s) selected`}
          <input
            className="landing__file-input"
            type="file"
            accept={ACCEPTED}
            multiple
            onChange={handleFiles}
          />
        </label>

        {files.length > 0 && (
          <ul className="landing__file-list">
            {files.map((f) => (
              <li key={f.name}>▸ {f.name}</li>
            ))}
          </ul>
        )}

        <button className="landing__submit" type="submit" disabled={files.length === 0}>
          Enter the Grimoire
        </button>

        {error && <p className="landing__error">{error}</p>}
      </form>
    </div>
  )
}
