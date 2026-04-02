// Loading screen shown while documents are being uploaded and the game world
// is generated. Displays a pulsing status message — no user interaction needed.
// Created: 2026-04-02
// Author: Devon Vanaenrode

interface Props {
  message?: string
}

export default function LoadingScreen({ message = 'Generating world...' }: Props) {
  return (
    <div className="screen">
      <p className="loading__text">{message}</p>
    </div>
  )
}
