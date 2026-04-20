// Results screen: shows overall score and a per-topic breakdown after a quiz session.
// "Quiz Again" returns to the quiz setup screen (same documents, score reset).
// "Upload New Documents" returns to the landing screen.
//
// Created: 2026-04-10
// Updated: 2026-04-10 — table wrapped in scrollable div for mobile
// Author: Devon Vanaenrode

import type { TopicScores } from '@/screens/QuizScreen'

interface Props {
  breakdown: TopicScores
  onQuizAgain: () => void
  onRestart: () => void
}

// Percentage rounded to the nearest integer, clamped 0–100
function pct(correct: number, total: number): number {
  if (total === 0) return 0
  return Math.round((correct / total) * 100)
}

export default function ResultsScreen({ breakdown, onQuizAgain, onRestart }: Props) {
  const topics = Object.entries(breakdown)

  const totalCorrect  = topics.reduce((s, [, t]) => s + t.correct, 0)
  const totalAnswered = topics.reduce((s, [, t]) => s + t.total,   0)
  const overallPct    = pct(totalCorrect, totalAnswered)

  return (
    <div className="screen">
      <h1 className="results__title">Results</h1>

      {/* Overall score */}
      <div className="results__overall">
        <span className="results__overall-score">{overallPct}%</span>
        <span className="results__overall-label">
          {totalCorrect} / {totalAnswered} correct
        </span>
      </div>

      {/* Per-topic breakdown — wrapper enables horizontal scroll on narrow screens */}
      {topics.length > 0 && (
        <div className="results__table-wrapper">
        <table className="results__table">
          <thead>
            <tr>
              <th className="results__th results__th--topic">Topic</th>
              <th className="results__th">Score</th>
              <th className="results__th">%</th>
            </tr>
          </thead>
          <tbody>
            {topics.map(([zone, { correct, total }]) => (
              <tr key={zone} className="results__row">
                <td className="results__td results__td--topic">{zone}</td>
                <td className="results__td">{correct} / {total}</td>
                <td className={`results__td results__td--pct ${pct(correct, total) >= 70 ? 'results__pct--good' : 'results__pct--low'}`}>
                  {pct(correct, total)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}

      <div className="results__actions">
        <button className="quiz__btn-primary" onClick={onQuizAgain}>
          Quiz Again
        </button>
        <button className="quiz__btn-ghost" onClick={onRestart}>
          Upload New Documents
        </button>
      </div>
    </div>
  )
}
