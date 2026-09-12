import { useState, useEffect } from 'react'
import type { Short } from '../types'
import { api } from '../api'
import './TranscriptView.css'

interface Props {
  project: string
  episode: string
  short: Short | null
}

export default function TranscriptView({ project, episode, short }: Props) {
  const [text, setText] = useState<string>('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!project || !episode || !short) return
    setLoading(true)
    api.getTranscriptText(project, episode)
      .then(setText)
      .catch(() => setText(''))
      .finally(() => setLoading(false))
  }, [project, episode, short])

  if (!short) return null

  return (
    <div className="transcript-view">
      <div className="transcript-header">
        <h3>Transcripción</h3>
      </div>
      <pre className="transcript-content">
        {loading ? 'Cargando...' : text || 'Sin transcripción disponible'}
      </pre>
    </div>
  )
}
