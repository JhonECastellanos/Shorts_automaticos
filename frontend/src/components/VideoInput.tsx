import { useState, useCallback, useEffect } from 'react'
import type { JobCreate, ModelStatus, CustomModelConfig } from '../types'
import { api } from '../api'
import './VideoInput.css'

interface Props {
  onSubmit: (data: JobCreate) => void
  disabled?: boolean
  current?: { project: string; episode: string } | null
}

const ALL_STEPS = [
  'download', 'transcribe', 'diarize', 'face_positions',
  'speaker_bind', 'analyze', 'edit', 'rank', 'export',
] as const

type Step = typeof ALL_STEPS[number]

const STEP_LABELS: Record<Step, string> = {
  download:       '01 Descarga',
  transcribe:     '02 Transcripción',
  diarize:        '03 Diarización',
  face_positions: '04 Posiciones faciales',
  speaker_bind:   '05 Voz ↔ rostro',
  analyze:        '06 Análisis IA',
  edit:           '07 Edición',
  rank:           '08 Ranking',
  export:         '09 Exportación',
}

const PROVIDER_ICONS: Record<string, string> = {
  gemini: '🚀',
  custom: '🔌',
}

export default function VideoInput({ onSubmit, disabled, current = null }: Props) {
  const [project, setProject] = useState('')
  const [episode, setEpisode] = useState('')
  const [url, setUrl] = useState('')
  const [mode, setMode] = useState<'url' | 'local'>('url')
  const [localFile, setLocalFile] = useState('')
  const [fromStep, setFromStep] = useState<Step>('download')
  const [aiModel, setAiModel] = useState('')
  const [confirming, setConfirming] = useState(false)
  const [showHint, setShowHint] = useState(false)

  // Modelos dinámicos desde el backend
  const [models, setModels] = useState<ModelStatus[]>([])
  const [defaultModel, setDefaultModel] = useState('')
  const [modelsLoading, setModelsLoading] = useState(true)

  // Custom model form
  const [showCustomForm, setShowCustomForm] = useState(false)
  const [customForm, setCustomForm] = useState<CustomModelConfig>({
    name: '', base_url: '', api_key: '', model_id: '', provider_type: 'openai_compatible',
  })
  const [customSaving, setCustomSaving] = useState(false)
  const [projectLimit, setProjectLimit] = useState<{ at_limit: boolean; count: number; max: number } | null>(null)

  // Sugerir nombre numérico automático (1..10) cuando no hay proyecto activo.
  useEffect(() => {
    if (current) return  // editando uno existente — no autonumerar
    let cancelled = false
    api.suggestProjectName()
      .then(res => {
        if (cancelled) return
        setProjectLimit({ at_limit: res.at_limit, count: res.current_count, max: res.max_projects })
        if (!project && res.suggested_name) setProject(res.suggested_name)
        if (!episode) setEpisode('ep01')
      })
      .catch(() => { /* silenciar — el backend puede no estar listo */ })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current])

  const loadModels = useCallback(() => {
    setModelsLoading(true)
    api.validateModels()
      .then(res => {
        setModels(res.models)
        setDefaultModel(res.default_model)
      })
      .catch(() => {})
      .finally(() => setModelsLoading(false))
  }, [])

  useEffect(() => { loadModels() }, [loadModels])

  const handleAddCustomModel = useCallback(async () => {
    if (!customForm.name || !customForm.base_url || !customForm.model_id) return
    setCustomSaving(true)
    try {
      await api.addCustomModel(customForm)
      setShowCustomForm(false)
      setCustomForm({ name: '', base_url: '', api_key: '', model_id: '', provider_type: 'openai_compatible' })
      loadModels()
    } catch (e) {
      alert(`Error: ${e instanceof Error ? e.message : e}`)
    } finally {
      setCustomSaving(false)
    }
  }, [customForm, loadModels])

  const handleDeleteCustomModel = useCallback(async (name: string) => {
    try {
      await api.deleteCustomModel(name)
      loadModels()
    } catch (e) {
      alert(`Error: ${e instanceof Error ? e.message : e}`)
    }
  }, [loadModels])

  const needsSource = fromStep === 'download'

  useEffect(() => {
    if (!current) return
    setProject(current.project)
    setEpisode(current.episode)
  }, [current])

  const disabledReason = (() => {
    if (disabled) return 'Procesando…'
    if (!project.trim()) return 'Falta el nombre del proyecto'
    if (!episode.trim()) return 'Falta el nombre del episodio'
    if (needsSource && mode === 'url' && !url.trim()) return 'Falta la URL de YouTube'
    if (needsSource && mode === 'local' && !localFile.trim()) return 'Falta la ruta del archivo local'
    // Bloquear creación de proyectos nuevos si ya se alcanzó el máximo (10).
    if (!current && projectLimit?.at_limit) {
      return `Límite alcanzado: ${projectLimit.count}/${projectLimit.max} proyectos. Elimina uno antes de crear otro.`
    }
    return ''
  })()
  const submitDisabled = Boolean(disabledReason)

  const performSubmit = useCallback(() => {
    setConfirming(false)
    if (!project.trim() || !episode.trim()) return

    const startIdx = ALL_STEPS.indexOf(fromStep)
    const steps = ALL_STEPS.slice(startIdx) as unknown as string[]

    const data: JobCreate = {
      project: project.trim(),
      episode: episode.trim(),
      steps,
      url: mode === 'url' ? url.trim() : undefined,
      local_file: mode === 'local' ? localFile.trim() : undefined,
      ai_model: aiModel || undefined,
    }

    if (!needsSource) {
      delete data.url
      delete data.local_file
    }

    onSubmit(data)
  }, [project, episode, fromStep, mode, url, localFile, aiModel, needsSource, onSubmit])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    // Si falta algo, mostrar el hint inline (no es spam — solo al intentar).
    if (disabledReason) {
      setShowHint(true)
      return
    }
    setShowHint(false)

    if (aiModel && current && !needsSource) {
      setConfirming(true)
      return
    }

    performSubmit()
  }, [disabledReason, aiModel, current, needsSource, performSubmit])

  // Ocultar hint automáticamente cuando el usuario corrige lo que faltaba.
  useEffect(() => {
    if (!disabledReason) setShowHint(false)
  }, [disabledReason])

  return (
    <div className="video-input-container">
      <form className="video-input-bar" onSubmit={handleSubmit}>
        <input
          className="vi-field vi-project"
          value={project}
          onChange={e => setProject(e.target.value)}
          placeholder="Proyecto"
          required
          title="Nombre del proyecto"
        />

        <input
          className="vi-field vi-episode"
          value={episode}
          onChange={e => setEpisode(e.target.value)}
          placeholder="ep01"
          required
          title="Nombre del episodio"
        />

        <select
          className="vi-from-step"
          value={fromStep}
          onChange={e => setFromStep(e.target.value as Step)}
          title="Iniciar desde este paso"
        >
          {ALL_STEPS.map(s => (
            <option key={s} value={s}>{STEP_LABELS[s]}</option>
          ))}
        </select>

        <select
          className="vi-ai-model"
          value={aiModel}
          onChange={e => setAiModel(e.target.value)}
          title="Modelo de IA para análisis"
        >
          <option value="">
            {modelsLoading ? 'Cargando modelos…' : `Por defecto (${defaultModel})`}
          </option>
          {models.map(m => (
            <option
              key={m.model}
              value={m.model}
              title={m.available ? `${m.response_time_ms}ms` : m.error || 'No disponible'}
            >
              {PROVIDER_ICONS[m.provider] || '❓'} {m.model}{!m.available ? ' ⚠️' : ''}
            </option>
          ))}
        </select>

        <button
          type="button"
          className="vi-add-model-btn"
          onClick={() => setShowCustomForm(v => !v)}
          title="Agregar modelo personalizado"
        >➕</button>

        {needsSource && (
          <>
            <div className="vi-mode-toggle">
              <button
                type="button"
                className={mode === 'url' ? 'active' : ''}
                onClick={() => setMode('url')}
                title="URL de YouTube"
              >🔗</button>
              <button
                type="button"
                className={mode === 'local' ? 'active' : ''}
                onClick={() => setMode('local')}
                title="Archivo local"
              >📁</button>
            </div>

            {mode === 'url' ? (
              <input
                className="vi-field vi-source"
                value={url}
                onChange={e => setUrl(e.target.value)}
                placeholder="https://youtu.be/..."
                type="text"
              />
            ) : (
              <input
                className="vi-field vi-source"
                value={localFile}
                onChange={e => setLocalFile(e.target.value)}
                placeholder="C:\\Videos\\ep01.mp4"
              />
            )}
          </>
        )}

        <button
          type="submit"
          className={`submit-btn${submitDisabled ? ' submit-btn--pending' : ''}`}
          disabled={disabled}
          title={disabledReason || (current ? 'Continuar proceso' : 'Iniciar proceso')}
        >
          {disabled ? 'Procesando...' : (current ? 'Continuar' : 'Iniciar')}
        </button>
      </form>
      {showHint && disabledReason && !disabled && (
        <div className="vi-hint" role="status">
          ⚠ {disabledReason}
        </div>
      )}

      {confirming && (
        <div className="custom-confirm-overlay">
          <div className="custom-confirm-modal">
            <div className="confirm-icon">⚠️</div>
            <h3>¿Reiniciar análisis con otro modelo?</h3>
            <p>
              Estás a punto de re-iniciar el proceso desde <strong>{fromStep}</strong> usando 
              el modelo <strong>{aiModel}</strong>.<br/><br/>
              ¡Esto eliminará la selección de shorts generada anteriormente y evaluará el video de nuevo!
            </p>
            <div className="confirm-actions">
              <button className="confirm-btn cancel" onClick={() => setConfirming(false)}>Cancelar</button>
              <button className="confirm-btn accept" onClick={performSubmit}>Aceptar y Reiniciar</button>
            </div>
          </div>
        </div>
      )}

      {showCustomForm && (
        <div className="custom-confirm-overlay">
          <div className="custom-confirm-modal" style={{ maxWidth: 420 }}>
            <h3>➕ Agregar modelo personalizado</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <input
                placeholder="Nombre (ej: Mi GPT-4o)"
                value={customForm.name}
                onChange={e => setCustomForm(f => ({ ...f, name: e.target.value }))}
                style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #555', background: '#2a2a2a', color: '#eee' }}
              />
              <input
                placeholder="URL base (ej: https://api.openai.com/v1)"
                value={customForm.base_url}
                onChange={e => setCustomForm(f => ({ ...f, base_url: e.target.value }))}
                style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #555', background: '#2a2a2a', color: '#eee' }}
              />
              <input
                placeholder="API Key (opcional)"
                type="password"
                value={customForm.api_key}
                onChange={e => setCustomForm(f => ({ ...f, api_key: e.target.value }))}
                style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #555', background: '#2a2a2a', color: '#eee' }}
              />
              <input
                placeholder="Model ID (ej: gpt-4o, llama3.2)"
                value={customForm.model_id}
                onChange={e => setCustomForm(f => ({ ...f, model_id: e.target.value }))}
                style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #555', background: '#2a2a2a', color: '#eee' }}
              />
              <select
                value={customForm.provider_type}
                onChange={e => setCustomForm(f => ({ ...f, provider_type: e.target.value }))}
                style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #555', background: '#2a2a2a', color: '#eee' }}
              >
                <option value="openai_compatible">OpenAI Compatible</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
            <div className="confirm-actions" style={{ marginTop: 12 }}>
              <button className="confirm-btn cancel" onClick={() => setShowCustomForm(false)}>Cancelar</button>
              <button
                className="confirm-btn accept"
                onClick={handleAddCustomModel}
                disabled={customSaving || !customForm.name || !customForm.base_url || !customForm.model_id}
              >
                {customSaving ? 'Guardando…' : 'Agregar'}
              </button>
            </div>
            {models.filter(m => m.provider === 'custom').length > 0 && (
              <div style={{ marginTop: 12, borderTop: '1px solid #444', paddingTop: 8 }}>
                <small style={{ color: '#999' }}>Modelos custom registrados:</small>
                {models.filter(m => m.provider === 'custom').map(m => (
                  <div key={m.model} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0' }}>
                    <span style={{ fontSize: 13 }}>🔌 {m.model} {m.available ? '✅' : '⛔'}</span>
                    <button
                      type="button"
                      onClick={() => handleDeleteCustomModel(m.model)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#f55', fontSize: 14 }}
                      title="Eliminar modelo"
                    >🗑</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
