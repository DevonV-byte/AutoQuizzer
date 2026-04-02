// Vite build configuration for the Grimoire game client.
// Configures the React plugin, resolves the @ path alias to src/, and
// proxies /api requests to the Python backend during development so the
// frontend and backend can run on separate ports without CORS issues.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(import.meta.dirname, 'src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        // Strip the /api prefix — backend routes are /upload, /health, etc.
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
