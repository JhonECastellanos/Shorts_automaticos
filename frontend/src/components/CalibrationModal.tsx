/**
 * Modal de calibración manual voz ↔ rostro.
 *
 * Aparece cuando `speaker_bind` detectó confianza baja en algún SPEAKER_xx.
 * El user ve thumbnails de las posiciones detectadas y para cada speaker
 * elige la posición que le corresponde. Al guardar, el binding se persiste
 * con `source: "manual"` y confidence=1.0.
 */
import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import './CalibrationModal.css'

interface Props {
  project: string
  episode: string
  onClose: () => void
  onSaved?: () => void
}

type Position = {
  position_id: number
  cx: number
  cy: number
  w: number
  h: number
  confidence: number
  thumb_url: string | null
}

type SpeakerRow = {
  speaker: string
  sample_start: number
  sample_end: number
  current_position_id: number | null
  confidence: number | null
  source: string | null
  needs_manual: boolean
}

export default function CalibrationModal({ project, episode, onClose, onSaved }: Props) {
  const [loading, setLoading] = useState(true)
  const [positions, setPositions] = useState<Position[]>([])
  const [speakers, setSpeakers] = useState<SpeakerRow[]>([])
  const [mapping, setMapping] = useState<Record<string, number>>({})
  const [saving, setSaving] = useState(false)
  const [playingSpeaker, setPlayingSpeaker] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getBindStatus(project, episode)
      setPositions(data.positions)
      setSpeakers(data.speakers)
      const initial: Record<string, number> = {}
      for (const s of data.speakers) {
        if (s.current_position_id !== null) initial[s.speaker] = s.current_position_id
      }
      setMapping(initial)
    } catch (e) {
      console.error('Error cargando bind status:', e)
    } finally {
      setLoading(false)
    }
  }, [project, episode])

  useEffect(() => { void load() }, [load])

  const handleSave = async () => {
    if (Object.keys(mapping).length === 0) return
    setSaving(true)
    try {
      await api.overrideBinding(project, episode, mapping)
      onSaved?.()
      onClose()
    } catch (e) {
      alert(`Error guardando: ${e instanceof Error ? e.message : e}`)
    } finally {
      setSaving(false)
    }
  }

  const playSample = (speaker: string, start: number, end: number) => {
    const audio = new Audio(`/api/videos/${project}/${episode}/stream#t=${start},${end}`)
    audio.currentTime = start
    setPlayingSpeaker(speaker)
    audio.addEventListener('timeupdate', () => {
      if (audio.currentTime >= end) { audio.pause(); setPlayingSpeaker(null) }
    })
    audio.addEventListener('ended', () => setPlayingSpeaker(null))
    void audio.play().catch(() => setPlayingSpeaker(null))
  }

  // Detectar colisiones (2 speakers asignados a la misma posición)
  const usedPositions = new Set<number>()
  const collisions = new Set<string>()
  for (const [sp, pos] of Object.entries(mapping)) {
    if (usedPositions.has(pos)) collisions.add(sp)
    usedPositions.add(pos)
  }

  return (
    <div className="custom-confirm-overlay" onClick={onClose}>
      <div className="custom-confirm-modal calibration-modal" onClick={e => e.stopPropagation()}>
        <div className="confirm-icon">🎯</div>
        <h3>Calibración: voz ↔ rostro</h3>
        <p style={{ marginBottom: 12 }}>
          Hicimos un binding automático con los primeros minutos del video, pero la confianza
          es baja para algún speaker. Escucha cada voz y elige el rostro correcto.
        </p>

        {loading && <p style={{ color: 'var(--text-muted)' }}>Cargando...</p>}

        {!loading && (
          <div className="calibration-grid">
            {speakers.map(sp => (
              <div key={sp.speaker} className={`calibration-row ${collisions.has(sp.speaker) ? 'calibration-collision' : ''}`}>
                <div className="calibration-speaker">
                  <div className="calibration-speaker-name">{sp.speaker}</div>
                  {sp.confidence !== null && (
                    <div className={`calibration-conf ${sp.needs_manual ? 'low' : 'ok'}`}>
                      {sp.needs_manual ? '⚠ baja' : '✓'} {Math.round((sp.confidence || 0) * 100)}%
                    </div>
                  )}
                  <button
                    className="calibration-play"
                    onClick={() => playSample(sp.speaker, sp.sample_start, sp.sample_end)}
                    disabled={playingSpeaker === sp.speaker}
                    type="button"
                  >
                    {playingSpeaker === sp.speaker ? '▶ ...' : '🔊 escuchar'}
                  </button>
                </div>

                <div className="calibration-positions">
                  {positions.map(pos => (
                    <button
                      key={pos.position_id}
                      type="button"
                      className={`calibration-thumb ${mapping[sp.speaker] === pos.position_id ? 'selected' : ''}`}
                      onClick={() => setMapping({ ...mapping, [sp.speaker]: pos.position_id })}
                    >
                      {pos.thumb_url ? (
                        <img src={pos.thumb_url} alt={`Posición ${pos.position_id}`} />
                      ) : (
                        <div className="calibration-thumb-placeholder">👤</div>
                      )}
                      <span className="calibration-thumb-label">#{pos.position_id}</span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {collisions.size > 0 && (
          <p className="calibration-warning">
            ⚠ Dos speakers tienen la misma posición asignada. Revisa antes de guardar.
          </p>
        )}

        <div className="confirm-actions">
          <button
            type="button"
            className="confirm-btn cancel"
            onClick={onClose}
            disabled={saving}
          >
            Cancelar
          </button>
          <button
            type="button"
            className="confirm-btn accept"
            onClick={handleSave}
            disabled={saving || Object.keys(mapping).length === 0 || collisions.size > 0}
          >
            {saving ? 'Guardando...' : 'Guardar calibración'}
          </button>
        </div>
      </div>
    </div>
  )
}
