import { useState, useCallback, useEffect, useMemo } from 'react'
import VideoInput from './components/VideoInput'
import ProcessingStatus from './components/ProcessingStatus'
import ShortsList from './components/ShortsList'
import VideoPlayer from './components/VideoPlayer'
import SettingsPanel from './components/SettingsPanel'
import ProjectSelector from './components/ProjectSelector'
import CalibrationModal from './components/CalibrationModal'
import ProcessingStatusHarness from './components/__ProcessingStatusHarness'
import { api } from './api'
import { useJob } from './hooks/useJob'
import { useSSE } from './hooks/useSSE'
import type { EpisodePipelineState, JobCreate, Short } from './types'
import './App.css'

export default function App() {
  // Dev harness: visitar /?harness=1 para probar transiciones de ProcessingStatus
  if (typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('harness') === '1') {
    return <ProcessingStatusHarness />
  }
  const { loading, error, createJob, refreshJobs, pauseJob, resumeJob, stopJob, cancelJob, getJobForProject, setJobs, patchJobLive } = useJob()
  const [selectedShort, setSelectedShort] = useState<Short | null>(null)
  // Restaurar el proyecto activo desde localStorage al recargar para no perder
  // contexto en un refresh del browser. El job de backend sigue corriendo en
  // paralelo — no se pausa al refrescar (eso requeriría acción explícita del user).
  const [activeProject, setActiveProject] = useState<{ project: string; episode: string } | null>(() => {
    try {
      const raw = typeof window !== 'undefined' ? window.localStorage.getItem('activeProject') : null
      if (!raw) return null
      const parsed = JSON.parse(raw)
      if (parsed?.project && parsed?.episode) return parsed
    } catch { /* ignore */ }
    return null
  })
  const [selectedPipelineState, setSelectedPipelineState] = useState<EpisodePipelineState | null>(null)

  // Sincronizar localStorage cuando cambia el proyecto activo
  useEffect(() => {
    try {
      if (activeProject) {
        window.localStorage.setItem('activeProject', JSON.stringify(activeProject))
      } else {
        window.localStorage.removeItem('activeProject')
      }
    } catch { /* quota / privacy mode — ignorar */ }
  }, [activeProject])

  // Al montar con un proyecto restaurado de localStorage, cargar su pipeline state.
  // Si el proyecto ya no existe (404), limpiamos la selección para evitar loops.
  useEffect(() => {
    if (activeProject && !selectedPipelineState) {
      void loadSelectedPipelineState(activeProject.project, activeProject.episode).then(result => {
        if (result === null) {
          // El proyecto no existe en el backend — limpiar selección
          setActiveProject(null)
        }
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const [projectsKey, setProjectsKey] = useState(0)
  const [exportProgress, setExportProgress] = useState<{ current: number; total: number } | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [controlBusy, setControlBusy] = useState(false)
  // postRankModalShownFor persistido en localStorage para que el modal solo
  // aparezca UNA vez por episodio (la primera que se generan borradores).
  // En reinicios posteriores del pipeline, el user puede usar el botón
  // "📦 Exportar" de ShortsList directamente.
  const [postRankModalShownFor, setPostRankModalShownFor] = useState<string | null>(() => {
    try { return window.localStorage.getItem('postRankModalShown') } catch { return null }
  })
  useEffect(() => {
    try {
      if (postRankModalShownFor) window.localStorage.setItem('postRankModalShown', postRankModalShownFor)
    } catch { /* ignore */ }
  }, [postRankModalShownFor])
  const [postRankModalOpen, setPostRankModalOpen] = useState(false)
  const [exportingAll, setExportingAll] = useState(false)
  const [calibModalOpen, setCalibModalOpen] = useState(false)
  const [calibCheckedFor, setCalibCheckedFor] = useState<string | null>(null)

  const activeJob = useMemo(() => {
    if (!activeProject) return null
    return getJobForProject(activeProject.project, activeProject.episode)
  }, [activeProject, getJobForProject])

  const visiblePipelineState = useMemo(() => {
    if (activeJob) {
      return {
        job_id: activeJob.id,
        project: activeJob.project,
        episode: activeJob.episode,
        steps: activeJob.steps,
        status: activeJob.status,
        current_step: activeJob.current_step,
        progress: activeJob.progress,
        steps_completed: activeJob.steps_completed,
        error: activeJob.error,
        created_at: activeJob.created_at,
        updated_at: activeJob.updated_at,
        started_at: activeJob.started_at,
        finished_at: activeJob.finished_at,
        elapsed_seconds: activeJob.elapsed_seconds,
        step_durations: activeJob.step_durations,
        can_pause: ['running', 'pending'].includes(activeJob.status),
        can_resume: ['paused', 'stopped', 'awaiting_export', 'failed', 'cancelled'].includes(activeJob.status),
        can_stop: ['running', 'pending', 'paused'].includes(activeJob.status),
        can_cancel: ['running', 'pending', 'paused', 'awaiting_export'].includes(activeJob.status),
      } satisfies EpisodePipelineState
    }
    return selectedPipelineState
  }, [activeJob, selectedPipelineState])

  // SSE solo para jobs activamente en ejecución — evita flood de requests para jobs terminados
  const liveStatuses = ['running', 'pending', 'paused']
  const sseJobId = (activeJob && liveStatuses.includes(activeJob.status))
    ? activeJob.id
    : (visiblePipelineState && liveStatuses.includes(visiblePipelineState.status)
        ? visiblePipelineState.job_id ?? null : null)
  // Para jobs terminados (completed, awaiting_export, failed, etc.),
  // cargar logs históricos sin conectar SSE
  const logsOnlyJobId = !sseJobId ? (visiblePipelineState?.job_id ?? null) : null
  const controlJobId = activeJob?.id ?? visiblePipelineState?.job_id ?? null
  const handleLiveEvent = useCallback((ev: import('./types').ProgressEvent) => {
    if (!ev.job_id || ev.step === 'done' || ev.step === 'awaiting_export') return
    const patch: Partial<import('./types').Job> = {
      current_step: ev.step,
      updated_at: new Date().toISOString(),
    }
    if (ev.percent !== null && ev.percent !== undefined) {
      patch.progress = ev.percent
    }
    patchJobLive(ev.job_id, patch)
  }, [patchJobLive])

  const { events, reset: resetEvents } = useSSE(sseJobId, logsOnlyJobId, {
    // Cada evento SSE en vivo actualiza el job en memoria AL INSTANTE.
    // Así la barra verde y el stepper reaccionan sin esperar al polling de 1.5s.
    // El override de `stableState` en ProcessingStatus se encarga de marcar
    // steps previos como 'done' cuando current_step avanza.
    onLiveEvent: handleLiveEvent,
  })

  const loadSelectedPipelineState = useCallback(async (project: string, episode: string) => {
    try {
      const next = await api.getPipelineStatus(project, episode)
      setSelectedPipelineState(next)
      return next
    } catch {
      setSelectedPipelineState(null)
      return null
    }
  }, [])

  const runControl = useCallback(async (fn: () => Promise<unknown>) => {
    if (controlBusy) return
    setControlBusy(true)
    try {
      resetEvents()
      await fn()
      await refreshJobs()
      if (activeProject) {
        void loadSelectedPipelineState(activeProject.project, activeProject.episode)
      }
      setProjectsKey(k => k + 1)
    } finally {
      setControlBusy(false)
    }
  }, [controlBusy, resetEvents, refreshJobs, activeProject, loadSelectedPipelineState])

  const handleSubmit = useCallback(async (data: JobCreate) => {
    resetEvents()
    setSelectedShort(null)
    try {
      const created = await createJob(data)
      if (created) {
        setActiveProject({ project: data.project, episode: data.episode })
        setSelectedPipelineState(null)
        setProjectsKey(k => k + 1)
        // Refresh inmediato para que `activeJob` refleje el NUEVO jobId y la
        // UI (barra + stepper) se resetee visualmente sin esperar al polling.
        await refreshJobs()
      } else {
        // createJob devolvió null → error ya está en useJob.error
        // Recargar pipeline state para no quedar en limbo
        if (activeProject) {
          void loadSelectedPipelineState(activeProject.project, activeProject.episode)
        }
      }
    } catch {
      if (activeProject) {
        void loadSelectedPipelineState(activeProject.project, activeProject.episode)
      }
    }
  }, [createJob, resetEvents, activeProject, loadSelectedPipelineState, refreshJobs])

  const handleProjectSelect = useCallback((project: string, episode: string) => {
    setActiveProject({ project, episode })
    setSelectedShort(null)
    void loadSelectedPipelineState(project, episode)
  }, [loadSelectedPipelineState])

  // Modal post-rank: se muestra UNA sola vez por episodio, cuando el pipeline
  // acaba de generar borradores frescos. Condiciones:
  //   - status = awaiting_export
  //   - steps_completed incluye 'rank' (no es un transitorio)
  //   - no se mostró antes para este project/episode (persistido en localStorage)
  // Reinicios posteriores NO vuelven a abrirlo — el user usa el botón
  // "📦 Exportar" de ShortsList que hace lo mismo.
  useEffect(() => {
    if (!activeJob) return
    const key = `${activeJob.project}/${activeJob.episode}`
    const rankDone = (activeJob.steps_completed ?? []).includes('rank')
    if (
      activeJob.status === 'awaiting_export'
      && rankDone
      && postRankModalShownFor !== key
    ) {
      setPostRankModalShownFor(key)
      setPostRankModalOpen(true)
    }
  }, [activeJob, postRankModalShownFor])

  // Abrir modal de calibración cuando:
  //   (a) el job actual acaba de terminar speaker_bind, O
  //   (b) se acaba de seleccionar un proyecto con binding de baja confianza.
  // Se muestra solo UNA vez por proyecto/episodio para no molestar.
  useEffect(() => {
    if (!activeProject) return
    const checkKey = `${activeProject.project}/${activeProject.episode}`
    if (calibCheckedFor === checkKey) return
    setCalibCheckedFor(checkKey)
    void api.getBindStatus(activeProject.project, activeProject.episode)
      .then(res => {
        if (res.needs_calibration) setCalibModalOpen(true)
      })
      .catch(() => { /* endpoint puede fallar si aún no hay binding */ })
  }, [activeProject, calibCheckedFor])

  const handleExportAll = useCallback(async () => {
    if (!activeProject || exportingAll) return
    setExportingAll(true)
    setPostRankModalOpen(false)
    try {
      const res = await api.exportAllShorts(activeProject.project, activeProject.episode)
      if (res.total) setExportProgress({ current: res.exported_count ?? res.total, total: res.total })
      await refreshJobs()
      setProjectsKey(k => k + 1)
    } catch (e) {
      alert(`Error exportando: ${e instanceof Error ? e.message : e}`)
    } finally {
      setExportingAll(false)
      setTimeout(() => setExportProgress(null), 1500)
    }
  }, [activeProject, exportingAll, refreshJobs])

  const handleProjectDeleted = useCallback((project: string, episode: string) => {
    if (activeProject?.project === project && activeProject?.episode === episode) {
      setActiveProject(null)
      setSelectedShort(null)
      setSelectedPipelineState(null)
    }
    setJobs(prev => prev.filter(job => !(job.project === project && job.episode === episode)))
    setProjectsKey(k => k + 1)
  }, [activeProject, setJobs])

  useEffect(() => {
    if (!activeProject || activeJob) return
    void loadSelectedPipelineState(activeProject.project, activeProject.episode)
  }, [activeProject, activeJob, loadSelectedPipelineState, projectsKey])

  useEffect(() => {
    // Solo procesar eventos de SSE en vivo — no históricos (logsOnlyJobId)
    if (!sseJobId || events.length === 0) return
    const last = events[events.length - 1]
    if (last.step === 'done') {
      void refreshJobs()
      if (activeProject) {
        void loadSelectedPipelineState(activeProject.project, activeProject.episode)
      }
      setProjectsKey(k => k + 1)
    } else {
      setJobs(prev => prev.map(job => {
        if (job.id !== last.job_id) return job

        const stepsCompleted = [...job.steps_completed]
        if (job.current_step && job.current_step !== last.step && !stepsCompleted.includes(job.current_step)) {
          stepsCompleted.push(job.current_step)
        }

        return {
          ...job,
          status: 'running',
          current_step: last.step,
          progress: last.percent ?? 0,
          steps_completed: stepsCompleted,
          updated_at: new Date().toISOString(),
        }
      }))
    }
  }, [events, sseJobId, refreshJobs, setJobs, activeProject, loadSelectedPipelineState])

  const hasJob = !!(visiblePipelineState || error)

  return (
    <div className="app">
      <main className="app-main">
        {/* ── Sidebar: solo proyectos ── */}
        <aside className="sidebar">
          <div className="sidebar-section-title">Proyectos</div>
          <ProjectSelector
            onSelect={handleProjectSelect}
            onDeleted={handleProjectDeleted}
            current={activeProject}
            refreshKey={projectsKey}
            onJobsChanged={() => { void refreshJobs() }}
          />
        </aside>

        {/* ── Contenido principal ── */}
        <section className="content">
          {activeProject ? (
            <>
              <div className="content-row">
                <div className="content-shorts">
                  <ShortsList
                    project={activeProject.project}
                    episode={activeProject.episode}
                    onSelect={setSelectedShort}
                    selected={selectedShort}
                    isAnalyzing={visiblePipelineState?.current_step === 'analyze' && visiblePipelineState?.status === 'running'}
                    onExportProgress={(current, total) => setExportProgress(current >= total ? null : { current, total })}
                    onOpenCalibration={() => setCalibModalOpen(true)}
                    canPreview={
                      // Los borradores solo existen tras completar el paso 'edit'.
                      // Permitimos preview si edit está en stepsCompleted o si el
                      // pipeline pasó a rank/awaiting_export/completed.
                      (visiblePipelineState?.steps_completed ?? []).includes('edit')
                      || ['awaiting_export', 'completed'].includes(visiblePipelineState?.status ?? '')
                      || (visiblePipelineState?.current_step === 'rank')
                    }
                  />
                </div>
                <div className="content-player">
                  <VideoPlayer project={activeProject.project} episode={activeProject.episode} short={selectedShort} hasActiveJob={!!activeJob || ['running', 'pending', 'paused'].includes(visiblePipelineState?.status ?? '')} />
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🎬</div>
              <h2>Sin proyecto activo</h2>
              <p>Selecciona un proyecto o inicia un nuevo procesamiento</p>
            </div>
          )}
        </section>
      </main>

      <div className="app-dock">
        <div className={`pipeline-strip${hasJob ? '' : ' pipeline-strip-hidden'}`}>
          {hasJob && (
            <ProcessingStatus
              // key force remount cuando cambia el jobId — reset visual limpio
              // (refs de stablePct/stableState frescos, barra desde 0).
              key={visiblePipelineState?.job_id ?? 'none'}
              jobId={visiblePipelineState?.job_id ?? undefined}
              status={visiblePipelineState?.status ?? 'pending'}
              currentStep={visiblePipelineState?.current_step ?? null}
              progress={visiblePipelineState?.progress ?? null}
              events={events}
              stepsCompleted={visiblePipelineState?.steps_completed ?? []}
              error={visiblePipelineState?.error ?? error}
              steps={visiblePipelineState?.steps ?? null}
              startedAt={visiblePipelineState?.started_at ?? null}
              elapsedSeconds={visiblePipelineState?.elapsed_seconds ?? null}
              stepDurations={visiblePipelineState?.step_durations ?? {}}
              onPause={controlJobId && visiblePipelineState?.can_pause && !controlBusy ? () => { void runControl(() => pauseJob(controlJobId!)) } : undefined}
              onResume={controlJobId && visiblePipelineState?.can_resume && !controlBusy ? () => { void runControl(() => resumeJob(controlJobId!)) } : undefined}
              onStop={controlJobId && visiblePipelineState?.can_stop && !controlBusy ? () => { void runControl(() => stopJob(controlJobId!)) } : undefined}
              onCancel={controlJobId && visiblePipelineState?.can_cancel && !controlBusy ? () => { void runControl(() => cancelJob(controlJobId!)) } : undefined}
              exportProgress={exportProgress}
            />
          )}
        </div>

        <header className="app-header">
          <div className="header-brand">
            <span className="header-icon">⚡</span>
            <h1>Script Editor</h1>
          </div>

          <div className="header-input">
            <VideoInput onSubmit={handleSubmit} disabled={loading} current={activeProject} />
          </div>

          {visiblePipelineState && (
            <span className={`header-badge badge ${
              visiblePipelineState.status === 'completed' ? 'success' :
              visiblePipelineState.status === 'failed' || visiblePipelineState.status === 'cancelled' ? 'error' :
              visiblePipelineState.status === 'awaiting_export' ? 'success' : 'info'
            }`}>{visiblePipelineState.status === 'awaiting_export' ? 'listo para exportar' : visiblePipelineState.status}</span>
          )}

          <button className="header-settings-btn" onClick={() => setSettingsOpen(true)} title="Configuración de procesamiento">
            ⚙️
          </button>
        </header>
      </div>

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* Modal calibración manual voz↔rostro (tras speaker_bind con baja confianza) */}
      {calibModalOpen && activeProject && (
        <CalibrationModal
          project={activeProject.project}
          episode={activeProject.episode}
          onClose={() => setCalibModalOpen(false)}
          onSaved={() => { setCalibModalOpen(false); void refreshJobs() }}
        />
      )}

      {/* Modal post-rank: se muestra al entrar en awaiting_export */}
      {postRankModalOpen && activeProject && (
        <div className="custom-confirm-overlay" onClick={() => setPostRankModalOpen(false)}>
          <div className="custom-confirm-modal" onClick={e => e.stopPropagation()}>
            <div className="confirm-icon">🎬</div>
            <h3>Borradores listos — ¿qué hacemos?</h3>
            <p>
              El pipeline generó los <b>borradores</b> (baja resolución) para que revises los cortes
              de cámara y afines rangos. Cuando estés listo, puedes exportar los shorts
              finales en <b>alta resolución</b>.
              <br/><br/>
              <b>Exportar todos</b>: renderiza en HD todos los shorts y distribuye a OCAMO.
              <br/>
              <b>Seleccionar</b>: cierra este aviso y elige manualmente cuáles exportar
              desde la lista.
            </p>
            <div className="confirm-actions">
              <button
                className="confirm-btn cancel"
                onClick={() => setPostRankModalOpen(false)}
              >
                Seleccionar manualmente
              </button>
              <button
                className="confirm-btn accept"
                onClick={handleExportAll}
                disabled={exportingAll}
              >
                {exportingAll ? '⏳ Exportando...' : 'Exportar todos'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
