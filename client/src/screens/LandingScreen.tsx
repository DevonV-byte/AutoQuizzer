// Landing screen: lets the user upload study documents to kick off the ingestion pipeline.
// Supports both click-to-browse and drag-and-drop file selection.
// Shows files currently in the database with per-file delete and a clear-all button.
// If files already exist, "Start Quiz" skips re-upload and goes straight to the quiz.
//
// Created: 2026-04-02
// Updated: 2026-04-20 — file list with per-file delete and clear-all; skip re-upload when
//                        DB already has files
// Author: Devon Vanaenrode

import { useState, useEffect } from 'react'
import { uploadDocuments, getZones, getFiles, deleteFile, clearAllFiles, ApiError } from '@/api/grimoire'
import type { FileEntry } from '@/api/grimoire'

interface Props {
  onLoading: () => void
  onReady: (zones: string[]) => void
  onError: (msg: string) => void
  error: string | null
}

// File types accepted by the backend ingestion pipeline
const ACCEPTED_EXTENSIONS = ['.txt', '.docx', '.ipynb', '.py']
const ACCEPTED = ACCEPTED_EXTENSIONS.join(',')

export default function LandingScreen({ onLoading, onReady, onError, error }: Props) {
  const [files,      setFiles]      = useState<File[]>([])
  const [dragging,   setDragging]   = useState(false)
  const [dbFiles,    setDbFiles]    = useState<FileEntry[]>([])
  const [fileError,  setFileError]  = useState<string | null>(null)

  // Load current DB contents on mount
  useEffect(() => {
    getFiles()
      .then(({ files }) => setDbFiles(files))
      .catch(() => setDbFiles([]))
  }, [])

  function handleFiles(e: React.ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(e.target.files ?? []))
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const dropped = Array.from(e.dataTransfer.files).filter((f) =>
      ACCEPTED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))
    )
    if (dropped.length > 0) setFiles(dropped)
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    setDragging(true)
  }

  function handleDragLeave(e: React.DragEvent) {
    if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false)
  }

  // Upload new files then proceed to quiz
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (files.length === 0) return
    onLoading()
    try {
      await uploadDocuments(files)
      const { zones } = await getZones()
      onReady(zones)
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Something went wrong.'
      onError(msg)
    }
  }

  // Skip upload and use the existing database
  async function handleStartWithExisting() {
    onLoading()
    try {
      const { zones } = await getZones()
      onReady(zones)
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Something went wrong.'
      onError(msg)
    }
  }

  async function handleDeleteFile(filename: string) {
    setFileError(null)
    try {
      await deleteFile(filename)
      setDbFiles((prev) => prev.filter((f) => f.filename !== filename))
    } catch (err) {
      setFileError(err instanceof ApiError ? err.detail : 'Failed to delete file.')
    }
  }

  async function handleClearAll() {
    setFileError(null)
    try {
      await clearAllFiles()
      setDbFiles([])
    } catch (err) {
      setFileError(err instanceof ApiError ? err.detail : 'Failed to clear database.')
    }
  }

  return (
    <div className="screen">
      <h1 className="landing__title">Grimoire</h1>
      <p className="landing__subtitle">Upload your notes. Get quizzed.</p>

      {/* Existing database section */}
      {dbFiles.length > 0 && (
        <div className="landing__db-section">
          <h2 className="landing__db-title">Currently in database</h2>
          <ul className="landing__db-list">
            {dbFiles.map((f) => (
              <li key={f.filename} className="landing__db-item">
                <span className="landing__db-filename">▸ {f.filename}</span>
                <button
                  className="landing__db-delete"
                  onClick={() => handleDeleteFile(f.filename)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>

          {fileError && <p className="landing__error">{fileError}</p>}

          <div className="landing__db-actions">
            <button className="landing__submit" onClick={handleStartWithExisting}>
              Start Quiz
            </button>
            <button className="landing__btn-ghost" onClick={handleClearAll}>
              Clear All
            </button>
          </div>
        </div>
      )}

      {/* Upload section */}
      <form className="landing__form" onSubmit={handleSubmit}>
        <p className="landing__upload-label">
          {dbFiles.length > 0 ? 'Add more documents:' : 'Upload your documents:'}
        </p>
        <label
          className={`landing__file-label ${dragging ? 'landing__file-label--dragging' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {files.length === 0
            ? 'Click to choose files or drag and drop'
            : `${files.length} file(s) selected`}
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
          {dbFiles.length > 0 ? 'Upload & Start Quiz' : 'Start Quiz'}
        </button>

        {error && <p className="landing__error">{error}</p>}
      </form>
    </div>
  )
}
