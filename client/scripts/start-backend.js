// Spawns the FastAPI backend (uvicorn) from the correct working directory.
// Using a Node script avoids cmd.exe / PowerShell shell-quoting differences
// when npm scripts try to cd out of the client/ subdirectory.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import { spawn } from 'child_process'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

// Run uvicorn from Code/ — .venv lives inside Code/, not the project root
const cwd = resolve(__dirname, '../..')
const python = resolve(cwd, '.venv/Scripts/python.exe')

const proc = spawn(python, ['-m', 'uvicorn', 'Backend.main:app', '--reload'], {
  cwd,
  stdio: 'inherit',
  shell: false,
})

proc.on('exit', (code) => process.exit(code ?? 0))
