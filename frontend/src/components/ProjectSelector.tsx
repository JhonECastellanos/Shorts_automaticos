import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import {
  formatPipelineDuration,
  getPipelineTiming,
} from '../pipeline'
import type { ProjectListItem } from '../types'
import './ProjectSelector.css'

interface Props {
  onSelect: (project: string, episode: string) => void
  onDeleted?: (project: string, episode: string) => void
  current: { project: string; episode: string } | null
  refreshKey?: number
  onJobsChanged?: () => void | Promise<void>
}

const STATUS_LABELS: Record<string, string> = {
  ready: 'Listo',
  pending: 'Pendiente',
  running: 'En curso',
  paused: 'Pausado',
  stopped: 'Detenido',
  completed: 'Completado',
  failed: 'Falló',
  cancelled: 'Cancelado',
  transcribed: 'Transcrito',
  awaiting_export: 'Listo para exportar',
}

export default function ProjectSelector({ onSelect, onDeleted, current, refreshKey, onJobsChanged: _onJobsChanged }: Props) {
  const [projects, setProjects] = useState<ProjectListItem[]>([])
  const [deletingKey, setDeletingKey] = useState<string | null>(null)
  const [nowMs, setNowMs] = useState(() => Date.now())

  const loadProjects = useCallback(async () => {
    try {
      setProjects(await api.listProjects())
    } catch {
      setProjects([])
    }
  }, [])

  useEffect(() => {
    void loadProjects()
  }, [loadProjects, refreshKey])

  const hasLiveProjects = projects.some(project => project.episodes.some(ep =>
    ['pending', 'running', 'paused'].includes(ep.status) && ep.job_id
  ))

  useEffect(() => {
    if (!hasLiveProjects) return

    const id = window.setInterval(() => {
      setNowMs(Date.now())
      void loadProjects()
    }, 3000)

    return () => window.clearInterval(id)
  }, [hasLiveProjects, loadProjects])

  const handleDelete = useCallback(async (project: string, episode: string) => {
    const confirmed = window.confirm(`Eliminar ${project} / ${episode}? Esta acción borra los datos internos y la carpeta pública.`)
    if (!confirmed) return

    const key = `${project}-${episode}`
    setDeletingKey(key)
    try {
      await api.deleteProject(project, episode)
      if (current?.project === project && current?.episode === episode) {
        onDeleted?.(project, episode)
      }
      await loadProjects()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'No se pudo eliminar el proyecto'
      window.alert(message)
    } finally {
      setDeletingKey(null)
    }
  }, [current, loadProjects, onDeleted])

  if (projects.length === 0) return (
    <p className="ps-empty">Sin proyectos aún.<br />Inicia un proceso para crear el primero.</p>
  )

  return (
    <div className="project-selector">
      {projects.map(p =>
        p.episodes.map(ep => {
          const isActive = current?.project === p.project && current?.episode === ep.name
          const cardKey = `${p.project}-${ep.name}`
          const deleteDisabled = deletingKey === cardKey || ['running', 'pending', 'paused'].includes(ep.status)
          const timing = getPipelineTiming({
            status: ep.status,
            steps: ep.steps,
            currentStep: ep.current_step,
            progress: ep.progress,
            stepsCompleted: ep.steps_completed,
            elapsedSeconds: ep.last_elapsed_seconds,
            startedAt: ep.started_at,
            stepDurations: ep.step_durations,
          }, nowMs)
          const progressValue = timing.overallProgress
          return (
            <div
              key={cardKey}
              className={`ps-card ${isActive ? 'active' : ''}`}
              onClick={() => onSelect(p.project, ep.name)}
              role="button"
              tabIndex={0}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') onSelect(p.project, ep.name) }}
            >
              <div className="ps-select">
                <div className="ps-top">
                  <span className="ps-project">{p.project}</span>
                  <span className="ps-ep">{ep.name}</span>
                  <span className={`ps-chip status status-${ep.status}`}>{STATUS_LABELS[ep.status] ?? ep.status}</span>
                </div>
                <div className="ps-bottom">
                  {['running', 'pending', 'paused'].includes(ep.status) && (
                    <span className="ps-chip timing">⌛ {formatPipelineDuration(timing.remainingSecs)}</span>
                  )}
                  <span className="ps-chip timing">{progressValue}%</span>
                </div>
              </div>
              <button
                type="button"
                className="ps-delete"
                disabled={deleteDisabled}
                title={deleteDisabled ? 'No se puede eliminar mientras está en ejecución' : 'Eliminar proyecto'}
                onClick={e => { e.stopPropagation(); void handleDelete(p.project, ep.name) }}
              >
                {deletingKey === cardKey ? '…' : '🗑'}
              </button>
            </div>
          )
        })
      )}
    </div>
  )
}
