import { useState, useEffect, useCallback } from 'react'
import { api } from '../api'
import type { HardwareInfo, ProcessingConfig } from '../types'
import './SettingsPanel.css'

interface Props {
  open: boolean
  onClose: () => void
}

export default function SettingsPanel({ open, onClose }: Props) {
  const [hw, setHw] = useState<HardwareInfo | null>(null)
  const [config, setConfig] = useState<ProcessingConfig>({ device: 'auto', ffmpeg_encoder: 'auto', ffmpeg_preset: 'auto' })
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [hwData, cfgData] = await Promise.all([
        api.getHardware(),
        api.getProcessingConfig(),
      ])
      setHw(hwData)
      setConfig(cfgData)
    } catch (e) {
      console.error('Error loading settings:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      void load()
      setSaved(false)
    }
  }, [open, load])

  const handleSave = async () => {
    setSaving(true)
    try {
      await api.updateProcessingConfig(config)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error('Error saving:', e)
    } finally {
      setSaving(false)
    }
  }

  if (!open) return null

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={e => e.stopPropagation()}>
        <div className="settings-header">
          <h2>⚙️ Configuración de Procesamiento</h2>
          <button className="settings-close" onClick={onClose}>✕</button>
        </div>

        {loading ? (
          <div className="settings-loading">Detectando hardware...</div>
        ) : (
          <>
            {/* Guía de la interfaz */}
            <div className="settings-section guide-section">
              <h3>📊 Guía de la Interfaz</h3>

              <details className="guide-group" open>
                <summary className="guide-summary">Scores de cada Short</summary>
                <ul className="guide-list">
                  <li><span className="guide-color" style={{ background: '#5b6ef5' }} /> <strong>IA</strong> — Puntuación base del modelo de IA. Evalúa calidad del contenido, relevancia y potencial viral <span className="guide-range">(0-10)</span></li>
                  <li><span className="guide-color" style={{ background: '#ffc107' }} /> <strong>Key</strong> — Keywords / trending. Mide presencia de palabras clave y temas tendencia <span className="guide-range">(0-7)</span></li>
                  <li><span className="guide-color" style={{ background: '#66bb6a' }} /> <strong>Hook</strong> — Gancho inicial. Evalúa qué tan atrapante es el inicio del short <span className="guide-range">(0-8)</span></li>
                  <li><span className="guide-color" style={{ background: '#ce93d8' }} /> <strong>Dur</strong> — Duración ideal. Penaliza shorts muy cortos (&lt;60s) o muy largos (&gt;180s) <span className="guide-range">(0-2)</span></li>
                </ul>
              </details>

              <details className="guide-group">
                <summary className="guide-summary">Colores de los Shorts (borde izquierdo)</summary>
                <ul className="guide-list">
                  <li>🔥 <strong>Dorado</strong> — Top 10% del ranking (mejores shorts)</li>
                  <li>⭐ <strong>Plateado</strong> — Top 30%</li>
                  <li>✨ <strong>Bronce</strong> — Top 50%</li>
                  <li>🔹 <strong>Normal</strong> — Resto del ranking</li>
                </ul>
              </details>

              <details className="guide-group">
                <summary className="guide-summary">Badge de duración (segundos)</summary>
                <ul className="guide-list">
                  <li><span className="guide-color dur-guide-ok" /> <strong>Verde</strong> — 60-180s (ideal para TikTok/Reels)</li>
                  <li><span className="guide-color dur-guide-warn" /> <strong>Amarillo</strong> — 45-59s (aceptable)</li>
                  <li><span className="guide-color dur-guide-bad" /> <strong>Rojo</strong> — Fuera de rango</li>
                </ul>
              </details>

              <details className="guide-group">
                <summary className="guide-summary">Trend Score (esquina derecha)</summary>
                <p className="guide-text">Combinación ponderada de IA + Key + Hook + Dur. Define el ranking final y la clasificación por tier del short.</p>
              </details>

              <details className="guide-group">
                <summary className="guide-summary">Pipeline — Iconos y porcentajes</summary>
                <ul className="guide-list guide-pipeline">
                  <li>⬇ <strong>Descarga</strong> — Video desde YouTube o archivo local</li>
                  <li>📝 <strong>Transcripción</strong> — Audio a texto con Whisper</li>
                  <li>👤 <strong>Detección facial</strong> — Posiciones de cara (MediaPipe/OpenCV)</li>
                  <li>📜 <strong>Guion</strong> — Asignación de texto a speakers con IA</li>
                  <li>✅ <strong>Validación</strong> — Coherencia speakers / face slots</li>
                  <li>🎙 <strong>Diarización</strong> — Mapeo temporal de speakers</li>
                  <li>🧠 <strong>Análisis</strong> — Detección de momentos con IA</li>
                  <li>🏆 <strong>Ranking</strong> — Ordenamiento y scoring final</li>
                  <li>🎬 <strong>Exportación</strong> — Videos, docs y FCPXML</li>
                </ul>
                <p className="guide-text">Cada paso muestra su progreso individual (0-100%). El progreso general es el promedio de los pasos completados. La exportación solo cuenta si se inicia manualmente tras el ranking.</p>
              </details>
            </div>

            {/* Hardware detected */}
            <div className="settings-section">
              <h3>Hardware Detectado</h3>
              <div className="hw-grid">
                <div className="hw-card">
                  <div className="hw-icon">🖥️</div>
                  <div className="hw-info">
                    <div className="hw-label">CPU</div>
                    <div className="hw-value">{hw?.cpu.name || 'No detectado'}</div>
                    <div className="hw-detail">{hw?.cpu.cores || 0} cores</div>
                  </div>
                </div>
                <div className={`hw-card ${hw?.gpu ? '' : 'hw-disabled'}`}>
                  <div className="hw-icon">🎮</div>
                  <div className="hw-info">
                    <div className="hw-label">GPU</div>
                    <div className="hw-value">{hw?.gpu?.name || 'No detectada'}</div>
                    {hw?.gpu && (
                      <div className="hw-detail">
                        {hw.gpu.vram_mb} MB VRAM · Compute {hw.gpu.compute_cap}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Capabilities */}
            <div className="settings-section">
              <h3>Capacidades</h3>
              <div className="cap-grid">
                <div className={`cap-item ${hw?.capabilities.nvenc ? 'cap-ok' : 'cap-no'}`}>
                  <span className="cap-icon">{hw?.capabilities.nvenc ? '✅' : '❌'}</span>
                  <span>NVENC (encoding GPU)</span>
                  {hw?.capabilities.nvenc && <span className="cap-badge">~2.5x más rápido</span>}
                </div>
                <div className={`cap-item ${hw?.capabilities.cuda_ai ? 'cap-ok' : 'cap-no'}`}>
                  <span className="cap-icon">{hw?.capabilities.cuda_ai ? '✅' : '❌'}</span>
                  <span>CUDA AI (Whisper GPU)</span>
                  {!hw?.capabilities.cuda_ai && hw?.gpu && (
                    <span className="cap-note">Requiere Compute ≥ 7.0</span>
                  )}
                </div>
              </div>
            </div>

            {/* Recommendations */}
            {hw?.recommendations && hw.recommendations.length > 0 && (
              <div className="settings-section">
                <h3>Recomendaciones</h3>
                <div className="rec-list">
                  {hw.recommendations.map((r, i) => (
                    <div key={i} className="rec-item">
                      <span className="rec-component">{r.component}</span>
                      <span className={`rec-badge rec-${r.recommendation}`}>{r.recommendation.toUpperCase()}</span>
                      <span className="rec-reason">{r.reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Config */}
            <div className="settings-section">
              <h3>Configuración</h3>
              <div className="config-grid">
                <label className="config-row">
                  <span className="config-label">Dispositivo</span>
                  <select
                    value={config.device}
                    onChange={e => setConfig(c => ({ ...c, device: e.target.value }))}
                  >
                    <option value="auto">Auto (recomendado)</option>
                    <option value="cpu">Solo CPU</option>
                    <option value="gpu">GPU (NVENC)</option>
                  </select>
                </label>

                <label className="config-row">
                  <span className="config-label">Encoder FFmpeg</span>
                  <select
                    value={config.ffmpeg_encoder}
                    onChange={e => setConfig(c => ({ ...c, ffmpeg_encoder: e.target.value }))}
                  >
                    <option value="auto">Auto</option>
                    <option value="libx264">libx264 (CPU)</option>
                    {hw?.capabilities.nvenc && (
                      <option value="h264_nvenc">h264_nvenc (GPU)</option>
                    )}
                  </select>
                </label>

                <label className="config-row">
                  <span className="config-label">Preset</span>
                  <select
                    value={config.ffmpeg_preset}
                    onChange={e => setConfig(c => ({ ...c, ffmpeg_preset: e.target.value }))}
                  >
                    <option value="auto">Auto</option>
                    {config.ffmpeg_encoder === 'h264_nvenc' ? (
                      <>
                        <option value="p1">p1 (más rápido)</option>
                        <option value="p4">p4 (balanceado)</option>
                        <option value="p7">p7 (mejor calidad)</option>
                      </>
                    ) : (
                      <>
                        <option value="ultrafast">ultrafast</option>
                        <option value="fast">fast</option>
                        <option value="medium">medium</option>
                        <option value="slow">slow</option>
                        <option value="veryslow">veryslow</option>
                      </>
                    )}
                  </select>
                </label>
              </div>
            </div>

            <div className="settings-footer">
              <button className="btn-save" onClick={handleSave} disabled={saving}>
                {saving ? 'Guardando...' : saved ? '✓ Guardado' : 'Guardar'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
