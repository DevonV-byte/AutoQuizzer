// Entry point for the Grimoire React application.
// Mounts the root App component into the #root DOM element.
// StrictMode is intentionally omitted: React's double-invoke behaviour in dev
// causes Phaser to mount twice and render two canvases.
// Created: 2026-04-02
// Author: Devon Vanaenrode

import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(<App />)
