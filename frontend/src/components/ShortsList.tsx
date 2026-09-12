import { useState, useEffect } from 'react'
import type { Short } from '../types'
import { api } from '../api'
import './ShortsList.css'

interface Props {
  project: string
  episode: string
  onSelect: (s: Short) => void
  selected: Short | null
  isAnalyzing?: boolean
  onExportProgress?: (current: number, total: number) => void
  /** Si true, los shorts son reproducibles (edit completado). Si false, se
   *  muestran pero deshabilitados con overlay explicativo. */
  canPreview?: boolean
  /** Abre el modal de calibración manual voz ↔ rostro. */
  onOpenCalibration?: () => void
}

type RankTier = 'gold' | 'silver' | 'bronze' | 'normal'

function getRankTier(rank: number, total: number): RankTier {
  if (total <= 0) return 'normal'
  const pct = rank / total
  if (pct <= 0.10) return 'gold'
  if (pct <= 0.30) return 'silver'
  if (pct <= 0.50) return 'bronze'
  return 'normal'
}

const TIER_ICON: Record<RankTier, string> = {
  gold: '🔥',
  silver: '⭐',
  bronze: '✨',
  normal: '',
}

function durationBadgeClass(dur: number): string {
  if (dur >= 60 && dur <= 180) return 'dur-ok'
  if (dur >= 45 && dur < 60) return 'dur-warn'
  return 'dur-bad'
}

export default function ShortsList({ project, episode, onSelect, selected, isAnalyzing, onExportProgress, canPreview = true, onOpenCalibration }: Props) {
  const [shorts, setShorts] = useState<Short[]>([])
  const [loading, setLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exportCount, setExportCount] = useState(0)
  const [exportTotal, setExportTotal] = useState(0)
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set())

  const loadShorts = () => {
    if (!project || !episode) return
    if (isAnalyzing) { setShorts([]); setSelectedIndices(new Set()); return }
    setLoading(true)
    api.listShorts(project, episode)
      .then(data => {
        setShorts(data)
        setSelectedIndices(new Set(data.map(s => s.rank ?? s.index)))
      })
      .catch(() => { setShorts([]); setSelectedIndices(new Set()) })
      .finally(() => setLoading(false))
  }

  // Re-cargar shorts cuando canPreview pasa a true (edit terminó) para refrescar
  // has_video/is_draft sin que el user tenga que recargar el browser.
  useEffect(loadShorts, [project, episode, isAnalyzing, canPreview])

  const handleDelete = async (e: React.MouseEvent, s: Short) => {
    e.stopPropagation()
    if (!confirm(`¿Eliminar short #${s.rank || s.index}?`)) return
    try {
      await api.deleteShort(project, episode, s.rank || s.index)
      setShorts(prev => prev.filter(x => x.index !== s.index))
      if (selected?.index === s.index) onSelect(null as unknown as Short)
    } catch { /* ignore */ }
  }

  const handleExportAll = async () => {
    const toExport = sortedShorts.filter(s => selectedIndices.has(s.rank ?? s.index))
    if (toExport.length === 0) return
    if (!confirm(`Se exportarán ${toExport.length} shorts seleccionados en alta resolución. Puede tomar varios minutos por short. ¿Continuar?`)) return
    const indices = toExport.map(s => s.rank || s.index)
    setExporting(true)
    setExportCount(0)
    setExportTotal(indices.length)
    onExportProgress?.(0, indices.length)
    try {
      const res = await api.exportShortsSelection(project, episode, indices)
      const done = res.exported_count ?? indices.length
      setExportCount(done)
      onExportProgress?.(done, indices.length)
      loadShorts()
    } catch (e) {
      alert(`Error exportando: ${e instanceof Error ? e.message : e}`)
    } finally {
      setExporting(false)
      // Delay para que la animación final llegue al 100%
      setTimeout(() => {
        setExportCount(0)
        setExportTotal(0)
        onExportProgress?.(indices.length, indices.length)
      }, 1500)
    }
  }

  const toggleSelect = (idx: number, e: React.MouseEvent | React.ChangeEvent) => {
    e.stopPropagation()
    setSelectedIndices(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const toggleAll = () => {
    if (selectedIndices.size === total) {
      setSelectedIndices(new Set())
    } else {
      setSelectedIndices(new Set(sortedShorts.map(s => s.rank ?? s.index)))
    }
  }

  // Ordenar por rank (ya vienen así del backend, pero asegurar)
  const sortedShorts = [...shorts].sort((a, b) => (a.rank ?? a.index) - (b.rank ?? b.index))
  const total = sortedShorts.length

  return (
    <div className="shorts-list">
      <div className="shorts-list-header">
        <h3>Shorts</h3>
        {total > 0 && (
          <span
            className="badge info shorts-select-badge"
            onClick={toggleAll}
            title={selectedIndices.size === total ? 'Deseleccionar todos' : 'Seleccionar todos'}
            style={{ cursor: 'pointer' }}
          >
            {selectedIndices.size}/{total}
          </span>
        )}
        {total > 0 && onOpenCalibration && (
          <button
            className="btn-calib"
            onClick={onOpenCalibration}
            title="Calibrar manualmente qué voz corresponde a cada rostro"
            type="button"
          >
            🎯 Calibrar
          </button>
        )}
        {total > 0 && (
          <button
            className="btn-export-all"
            onClick={handleExportAll}
            disabled={exporting || selectedIndices.size === 0 || !canPreview}
            title={!canPreview
              ? 'Los borradores aún se están generando en el paso de Edición'
              : `Exportar ${selectedIndices.size} shorts seleccionados`}
          >
            {exporting ? `⏳ ${exportCount}/${exportTotal}` : `📦 Exportar (${selectedIndices.size})`}
          </button>
        )}
      </div>
      {total > 0 && !canPreview && (
        <div className="shorts-gating-banner">
          ⚠ Los borradores (video 9:16 con cortes de cámara y subtítulos) estarán disponibles cuando el paso <b>Edición</b> llegue al 100%.
        </div>
      )}
      <div className="shorts-body">
        {/* Leyenda de tiers */}
        {!isAnalyzing && total > 0 && (
          <div className="tier-legend">
            <span className="legend-item"><span className="tier-badge">🔥</span> Top 10%</span>
            <span className="legend-item"><span className="tier-badge">⭐</span> Top 30%</span>
            <span className="legend-item"><span className="tier-badge">✨</span> Top 50%</span>
          </div>
        )}
        {isAnalyzing && (
          <div className="analyzing-state">
            <span>🧠</span>
            <p style={{ color: 'var(--text-muted)' }}>La IA está evaluando la transcripción para seleccionar 35 shorts de 60-180s.</p>
          </div>
        )}
        {!isAnalyzing && loading && <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Cargando...</p>}
        {!isAnalyzing && !loading && total === 0 && (
          <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Sin shorts detectados</p>
        )}
        {!isAnalyzing && sortedShorts.map(s => {
          const rank = s.rank ?? s.index
          const tier = getRankTier(rank, total)
          const tierIcon = TIER_ICON[tier]
          const isSelected = selectedIndices.has(rank)
          const playable = canPreview && (s.has_video || true) // has_video se actualiza tras edit
          return (
            <div
              key={s.index}
              className={`short-card tier-${tier} ${selected?.index === s.index ? 'selected' : ''} ${!isSelected ? 'deselected' : ''} ${!playable ? 'not-playable' : ''}`}
              onClick={() => { if (playable) onSelect(s) }}
              title={!playable ? 'Borrador aún no generado. Espera a que el paso Edición llegue al 100%.' : undefined}
            >
              <div className="short-header">
                <input
                  type="checkbox"
                  className="short-checkbox"
                  checked={isSelected}
                  onChange={(e) => toggleSelect(rank, e)}
                  onClick={(e) => e.stopPropagation()}
                />
                <span className="short-rank">
                  {tierIcon && <span className="tier-badge">{tierIcon}</span>}
                  #{rank}
                </span>
                {s.has_video && !s.is_draft && <span className="badge success">Final</span>}
                {s.has_video && s.is_draft && <span className="badge draft">Borrador</span>}
                {s.trend_score !== null && (
                  <span className={`trend-score trend-${tier}`}>{s.trend_score}</span>
                )}
                <button
                  className="btn-delete-short"
                  onClick={(e) => handleDelete(e, s)}
                  title="Eliminar short"
                >🗑️</button>
              </div>
              <div className="short-topic">{s.topic || 'Sin tema'}</div>
              <div className="short-meta">
                <span className={`dur-badge ${durationBadgeClass(s.duration)}`}>{s.duration.toFixed(0)}s</span>
                {s.score !== null && <span>IA: {s.score}</span>}
                {s.dominant_speaker && <span className="speaker-tag">{s.dominant_speaker}</span>}
              </div>
              {s.hook && <div className="short-hook">"{s.hook}"</div>}
              {/* Score breakdown bars */}
              {s.trend_score !== null && s.base_score !== null && (
                <div className="score-breakdown">
                  <div className="score-bar-row">
                    <span className="score-label">IA</span>
                    <div className="score-bar"><div className="score-fill fill-base" style={{ width: `${Math.min(100, (s.base_score || 0) * 10)}%` }} /></div>
                  </div>
                  <div className="score-bar-row">
                    <span className="score-label">Key</span>
                    <div className="score-bar"><div className="score-fill fill-keyword" style={{ width: `${Math.min(100, (s.keyword_score || 0) * 15)}%` }} /></div>
                  </div>
                  <div className="score-bar-row">
                    <span className="score-label">Hook</span>
                    <div className="score-bar"><div className="score-fill fill-hook" style={{ width: `${Math.min(100, (s.hook_score || 0) * 12)}%` }} /></div>
                  </div>
                  <div className="score-bar-row">
                    <span className="score-label">Dur</span>
                    <div className="score-bar"><div className="score-fill fill-dur" style={{ width: `${Math.min(100, (s.duration_score || 0) * 50)}%` }} /></div>
                  </div>
                </div>
              )}
              {s.categories && s.categories.length > 0 && (
                <div className="short-tags">
                  {s.categories.map(c => (
                    <span key={c} className="tag">{c}</span>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
