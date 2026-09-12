import { useState, useEffect, useRef } from 'react'
import type { Short } from '../types'
import { api } from '../api'
import './VideoPlayer.css'

interface Props {
  project: string
  episode: string
  short: Short | null
  /** Si hay un job activo (running/pending/paused), el probe reintenta en 404.
   *  Si no hay job activo y el video no existe aún, se detiene para no saturar logs. */
  hasActiveJob?: boolean
}

interface SubConfig {
  font: string
  color: string
  size: 'S' | 'M' | 'L'
  outline: boolean
}

const FONTS = [
  { id: 'Arial',       label: 'Arial' },
  { id: 'Impact',      label: 'Impact' },
  { id: 'Oswald',      label: 'Oswald' },
  { id: 'Bebas Neue',  label: 'Bebas' },
]

const COLORS = [
  { id: 'white',   label: 'Blanco',  hex: '#ffffff' },
  { id: 'yellow',  label: 'Amarillo', hex: '#ffe800' },
  { id: '#00ff99', label: 'Verde',    hex: '#00ff99' },
  { id: '#00e5ff', label: 'Cian',     hex: '#00e5ff' },
]

function fmtTime(secs: number): string {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function VideoPlayer({ project, episode, short: s, hasActiveJob = false }: Props) {
  const [sub, setSub] = useState<SubConfig>({
    font: 'Arial', color: 'white', size: 'M', outline: true
  })
  const [showSubConfig, setShowSubConfig] = useState(false)
  const [videoSrc, setVideoSrc] = useState<string | null>(null)
  
  // Time Editor State
  const [tStart, setTStart] = useState<number>(0)
  const [tEnd, setTEnd] = useState<number>(0)
  const [duration, setDuration] = useState<number>(0)
  const [saving, setSaving] = useState(false)
  const [exportingShort, setExportingShort] = useState(false)
  const [previewingShort, setPreviewingShort] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewExpanded, setPreviewExpanded] = useState(false)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const videoRef = useRef<HTMLVideoElement>(null)
  const rafRef = useRef<number | null>(null)
  const playheadRef = useRef<HTMLDivElement>(null)
  const tStartRef = useRef(tStart)
  const tEndRef = useRef(tEnd)
  const durationRef = useRef(duration)
  tStartRef.current = tStart
  tEndRef.current = tEnd
  durationRef.current = duration

  // Usar el endpoint de streaming. Pre-verificamos vía el endpoint VideoInfo
  // (GET /api/videos/{p}/{e}) que existe ligero y retorna 404 si aún no hay
  // video descargado. Evitamos que el <video src> dispare un 404 ruidoso en
  // la consola cuando la descarga aún está en curso. Si el video no existe
  // Y hay un job activo, reintentamos cada 15s.
  const hasActiveJobRef = useRef(hasActiveJob)
  hasActiveJobRef.current = hasActiveJob

  useEffect(() => {
    if (!project || !episode) {
      setVideoSrc(null)
      return
    }
    const streamUrl = `/api/videos/${project}/${episode}/stream`
    const probeUrl = `/api/videos/${encodeURIComponent(project)}/${encodeURIComponent(episode)}`
    let cancelled = false
    let retryHandle: ReturnType<typeof setTimeout> | null = null

    const probe = async () => {
      if (cancelled) return
      try {
        const res = await fetch(probeUrl)
        if (cancelled) return
        if (res.ok) {
          setVideoSrc(streamUrl)
          return
        }
        // 404 definitivo sin job activo → no reintentar (evita flood de logs)
        if (res.status === 404 && !hasActiveJobRef.current) {
          setVideoSrc(null)
          return
        }
      } catch { /* red caída — reintentar */ }
      if (cancelled) return
      setVideoSrc(null)
      // 15s entre reintentos — suficiente para que el user vea la respuesta cuando termine
      // download, sin saturar los logs del backend con probes durante 1+ hora de descarga.
      retryHandle = setTimeout(probe, 15000)
    }
    void probe()
    return () => {
      cancelled = true
      if (retryHandle) clearTimeout(retryHandle)
    }
  }, [project, episode])

  useEffect(() => {
    if (s) {
      setTStart(s.start)
      setTEnd(s.end)
      setPreviewUrl(null)
      setPreviewExpanded(false)
      if (videoRef.current) {
        videoRef.current.currentTime = s.start
      }
    }
  }, [s])

  // RAF loop: actualiza playhead via DOM directo (sin re-render), sync state solo en pause/seek
  useEffect(() => {
    if (!s || !videoSrc || !videoRef.current) return

    const v = videoRef.current

    const updatePlayhead = (ct: number) => {
      if (!playheadRef.current) return
      const dur = durationRef.current
      const pct = dur > 0 ? Math.max(0, Math.min(100, (ct / dur) * 100)) : 0
      playheadRef.current.style.left = `${pct}%`
    }

    const tick = () => {
      if (v.paused || v.ended) {
        rafRef.current = null
        return
      }
      const ct = v.currentTime
      updatePlayhead(ct)
      if (ct > tEndRef.current) {
        v.pause()
        v.currentTime = tStartRef.current
        setCurrentTime(tStartRef.current)
        rafRef.current = null
        return
      }
      rafRef.current = requestAnimationFrame(tick)
    }

    const onPlay = () => {
      if (v.currentTime < tStartRef.current || v.currentTime > tEndRef.current) {
        v.currentTime = tStartRef.current
        setCurrentTime(tStartRef.current)
      }
      if (!rafRef.current) rafRef.current = requestAnimationFrame(tick)
    }
    const onPause = () => { setCurrentTime(v.currentTime) }
    const onSeeked = () => { setCurrentTime(v.currentTime) }

    v.addEventListener('play', onPlay)
    v.addEventListener('pause', onPause)
    v.addEventListener('seeked', onSeeked)

    if (!v.paused) onPlay()

    return () => {
      if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
      v.removeEventListener('play', onPlay)
      v.removeEventListener('pause', onPause)
      v.removeEventListener('seeked', onSeeked)
    }
  }, [s, videoSrc])

  const handleSaveTime = async () => {
    if (!s || saving) return
    setSaving(true)
    try {
      await api.updateShortTime(project, episode, s.index, tStart, tEnd)
      alert('Tiempos guardados exitosamente. Recuerda exportar para generar el video cortado.')
    } catch {
      alert('Error guardando los tiempos')
    } finally {
      setSaving(false)
    }
  }

  const handleExportShort = async () => {
    if (!s || exportingShort) return
    setExportingShort(true)
    try {
      await api.exportShort(project, episode, s.rank || s.index)
      alert('Short exportado exitosamente.')
    } catch {
      alert('Error exportando el short')
    } finally {
      setExportingShort(false)
    }
  }

  const handlePreview = async () => {
    if (!s || previewingShort) return
    setPreviewingShort(true)
    setPreviewExpanded(true)
    try {
      const res = await api.previewShort(project, episode, s.rank || s.index)
      void res
      const num = s.rank || s.index
      setPreviewUrl(`/projects/${project}/${episode}/output/short_${String(num).padStart(2, '0')}/preview_${String(num).padStart(2, '0')}.mp4?t=${Date.now()}`)
    } catch {
      alert('Error generando preview')
    } finally {
      setPreviewingShort(false)
    }
  }

  const shiftStart = (amt: number) => setTStart(p => Math.max(0, p + amt))
  const shiftEnd = (amt: number) => setTEnd(p => Math.max(tStart + 1, p + amt))

  const showExportedVideo = s && s.has_video && s.file

  return (
    <div className="video-player">
      {/* ── Zona de video (siempre visible) ── */}
      <div className="player-video-wrap">
        {previewUrl ? (
          <>
            <video controls className="player-video" key={previewUrl} autoPlay>
              <source src={previewUrl} type="video/mp4" />
            </video>
            <div className="preview-badge draft-badge">
              BORRADOR — con subtítulos y cambio de cámara
              <button className="draft-close" onClick={() => { setPreviewUrl(null); setPreviewExpanded(false); }}>✕</button>
            </div>
          </>
        ) : showExportedVideo ? (
          <video controls className="player-video" key={s.file} autoPlay>
            <source src={s.file!} type="video/mp4" />
          </video>
        ) : videoSrc ? (
          <>
            <video 
              ref={videoRef} 
              controls 
              preload="metadata"
              className="player-video" 
              key={videoSrc}
              onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
            >
              <source src={videoSrc} type="video/mp4" />
            </video>
            {s && (
              <div className="preview-badge">
                PREVIEW: {fmtTime(tStart)} → {fmtTime(tEnd)}
              </div>
            )}
          </>
        ) : (
          <div className="player-no-video">
            <span>Cargando video...</span>
          </div>
        )}
      </div>

      {/* ── Full timeline context (minimap) ── */}
      {!showExportedVideo && duration > 0 && s && (
        <div className="custom-timeline-wrap mini">
          <div className="custom-timeline-track mini-track">
            <div 
              className={`custom-timeline-range${previewExpanded ? ' expanded' : ''}`}
              style={{
                left: previewExpanded ? '0%' : `${Math.max(0, Math.min(100, (tStart / duration) * 100))}%`,
                width: previewExpanded ? '100%' : `${Math.max(0.5, Math.min(100, ((tEnd - tStart) / duration) * 100))}%`
              }}
            />
            {!previewExpanded && (
              <div
                ref={playheadRef}
                className="timeline-playhead global"
                style={{
                  left: `${Math.max(0, Math.min(100, (currentTime / duration) * 100))}%`
                }}
              />
            )}
          </div>
        </div>
      )}

      {/* ── Panel inferior (siempre visible para mantener el layout) ── */}
      <div className="player-bottom-panel">
        {!s ? (
          <div className="player-bottom-empty">
            <span>Selecciona un short en la lista para editar sus tiempos y subtítulos</span>
          </div>
        ) : (
          <>
            {/* Info del short */}
            <div className="player-info-row">
              <div className="player-info-detail">
                <span className="player-info-title">#{s.rank || s.index} — {s.topic || 'Sin tema'}</span>
                {s.hook && <span className="detail-hook">{s.hook}</span>}
              </div>
              <span className="time-duration-badge">{(tEnd - tStart).toFixed(1)}s</span>
            </div>

            {/* Editor de Tiempo */}
            <div className="time-editor">
              <div className="time-editor-col">
                <label>Inicio</label>
                <div className="time-editor-controls">
                  <button disabled={saving} onClick={() => shiftStart(-0.5)}>−</button>
                  <input 
                    type="number" 
                    value={Number(tStart).toFixed(1)} 
                    step="0.1"
                    onChange={e => setTStart(parseFloat(e.target.value) || 0)}
                  />
                  <button disabled={saving} onClick={() => shiftStart(0.5)}>+</button>
                </div>
              </div>
              <div className="time-editor-col">
                <label>Fin</label>
                <div className="time-editor-controls">
                  <button disabled={saving} onClick={() => shiftEnd(-0.5)}>−</button>
                  <input 
                    type="number" 
                    value={Number(tEnd).toFixed(1)} 
                    step="0.1"
                    onChange={e => setTEnd(parseFloat(e.target.value) || 0)}
                  />
                  <button disabled={saving} onClick={() => shiftEnd(0.5)}>+</button>
                </div>
              </div>
              <button 
                className="time-save-btn" 
                disabled={saving || (tStart === s.start && tEnd === s.end)} 
                onClick={handleSaveTime}
              >
                {saving ? '...' : '💾 Guardar'}
              </button>

              <button
                className="time-save-btn"
                disabled={exportingShort}
                onClick={handleExportShort}
                title="Exportar este short como video"
              >
                {exportingShort ? '⏳' : '📦'} Exportar
              </button>

              <button
                className="time-save-btn preview-btn"
                disabled={previewingShort}
                onClick={handlePreview}
                title="Generar video borrador rápido (baja calidad)"
              >
                {previewingShort ? '⏳' : '👁️'} Preview
              </button>

              {/* Toggle de subtítulos */}
              <button 
                className={`sub-toggle-btn ${showSubConfig ? 'active' : ''}`}
                onClick={() => setShowSubConfig(!showSubConfig)}
                title="Configurar Subtítulos"
                type="button"
              >
                CC
              </button>
            </div>

            {/* Desplegable de subtítulos — no mueve el layout (absolute/fixed overlay) */}
            {showSubConfig && (
              <div className="sub-config-inline">
                <div className="sub-config-section">
                  <span className="sub-config-label">Fuente</span>
                  <div className="sub-options">
                    {FONTS.map(f => (
                      <button
                        key={f.id}
                        type="button"
                        className={`sub-opt-btn${sub.font === f.id ? ' active' : ''}`}
                        style={{ fontFamily: f.id }}
                        onClick={() => setSub(p => ({ ...p, font: f.id }))}
                      >{f.label}</button>
                    ))}
                  </div>
                </div>
                <div className="sub-config-section">
                  <span className="sub-config-label">Color</span>
                  <div className="sub-options">
                    {COLORS.map(c => (
                      <button
                        key={c.id}
                        type="button"
                        className={`sub-color-btn${sub.color === c.id ? ' active' : ''}`}
                        style={{ '--swatch': c.hex } as React.CSSProperties}
                        onClick={() => setSub(p => ({ ...p, color: c.id }))}
                        title={c.label}
                      />
                    ))}
                  </div>
                </div>
                <div className="sub-config-section">
                  <span className="sub-config-label">Tamaño</span>
                  <div className="sub-options">
                    {(['S','M','L'] as const).map(sz => (
                      <button
                        key={sz}
                        type="button"
                        className={`sub-opt-btn sub-sz${sub.size === sz ? ' active' : ''}`}
                        onClick={() => setSub(p => ({ ...p, size: sz }))}
                      >{sz}</button>
                    ))}
                    <label className="sub-outline-toggle">
                      <input
                        type="checkbox"
                        checked={sub.outline}
                        onChange={e => setSub(p => ({ ...p, outline: e.target.checked }))}
                      />
                      <span>Borde</span>
                    </label>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
