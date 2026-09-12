import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { api } from '../api'
import type { Job, JobCreate } from '../types'

const LIVE_STATUSES = new Set<Job['status']>(['pending', 'running', 'paused'])

/** Statuses that represent a job the frontend should treat as "active". */
const ACTIVE_STATUSES = new Set<Job['status']>([
  'pending', 'running', 'paused', 'stopped', 'awaiting_export',
])

function sortJobs(jobs: Job[]): Job[] {
  const statusPriority: Record<Job['status'], number> = {
    running: 0,
    paused: 1,
    pending: 2,
    awaiting_export: 3,
    stopped: 4,
    failed: 5,
    completed: 6,
    cancelled: 7,
  }

  return [...jobs].sort((left, right) => {
    const byStatus = statusPriority[left.status] - statusPriority[right.status]
    if (byStatus !== 0) return byStatus

    const leftTime = Date.parse(left.updated_at || left.created_at || '') || 0
    const rightTime = Date.parse(right.updated_at || right.created_at || '') || 0
    if (rightTime !== leftTime) return rightTime - leftTime

    return `${left.project}/${left.episode}`.localeCompare(`${right.project}/${right.episode}`)
  })
}

export function useJob() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mergeJob = useCallback((job: Job) => {
    setJobs(prev => sortJobs([...prev.filter(current => current.id !== job.id), job]))
  }, [])

  const refreshJobs = useCallback(async () => {
    try {
      const updated = await api.listJobs()
      setJobs(prev => {
        // Avoid unnecessary state update if data hasn't changed
        const prevIds = prev.map(j => `${j.id}:${j.status}:${j.current_step}:${j.progress}:${j.updated_at}`).join('|')
        const nextSorted = sortJobs(updated)
        const nextIds = nextSorted.map(j => `${j.id}:${j.status}:${j.current_step}:${j.progress}:${j.updated_at}`).join('|')
        return prevIds === nextIds ? prev : nextSorted
      })
      return updated
    } catch {
      return []
    }
  }, [])

  useEffect(() => {
    void refreshJobs()
  }, [refreshJobs])

  // Derive a stable boolean for polling — avoids effect teardown/setup on every poll
  const hasLiveJobs = useMemo(
    () => jobs.some(job => LIVE_STATUSES.has(job.status)),
    [jobs],
  )
  const refreshRef = useRef(refreshJobs)
  refreshRef.current = refreshJobs

  useEffect(() => {
    if (!hasLiveJobs) return

    // Polling cada 1.5s mientras hay live jobs. El SSE aporta actualizaciones
    // entre polls para fluidez visual; el polling es safety net y fuente de
    // verdad para stepsCompleted/step_durations.
    const id = window.setInterval(() => {
      void refreshRef.current()
    }, 1500)

    return () => window.clearInterval(id)
  }, [hasLiveJobs])

  /**
   * Merge parcial de un job existente (para actualizaciones optimistas
   * desde SSE). No dispara network call; solo actualiza el state local.
   */
  const patchJobLive = useCallback((jobId: string, patch: Partial<Job>) => {
    setJobs(prev => prev.map(j => (j.id === jobId ? { ...j, ...patch } : j)))
  }, [])

  const createJob = useCallback(async (data: JobCreate) => {
    setLoading(true)
    setError(null)
    try {
      const created = await api.createJob(data)
      mergeJob(created)
      return created
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Error desconocido'
      setError(msg)
      return null
    } finally {
      setLoading(false)
    }
  }, [mergeJob])

  const pauseJob = useCallback(async (jobId: string) => {
    try {
      const updated = await api.pauseJob(jobId)
      mergeJob(updated)
      return updated
    } catch { /* ignore */ }
    return null
  }, [mergeJob])

  const resumeJob = useCallback(async (jobId: string) => {
    try {
      const updated = await api.resumeJob(jobId)
      mergeJob(updated)
      return updated
    } catch { /* ignore */ }
    return null
  }, [mergeJob])

  const stopJob = useCallback(async (jobId: string) => {
    try {
      const updated = await api.stopJob(jobId)
      mergeJob(updated)
      return updated
    } catch { /* ignore */ }
    return null
  }, [mergeJob])

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      await api.cancelJob(jobId)
      const updated = await api.getJob(jobId)
      mergeJob(updated)
      return updated
    } catch { /* ignore */ }
    return null
  }, [mergeJob])

  const getJobForProject = useCallback((project: string, episode: string) => {
    // Case-insensitive match so "Ocamo1" and "OCAMO1" find the same project
    const lowerProject = project.toLowerCase()
    const lowerEpisode = episode.toLowerCase()
    const matching = jobs.filter(
      job => job.project.toLowerCase() === lowerProject
        && job.episode.toLowerCase() === lowerEpisode,
    )
    if (matching.length === 0) return null

    // Pick the most recently updated job
    const newest = [...matching].sort((a, b) => {
      const ta = Date.parse(a.updated_at || a.created_at || '') || 0
      const tb = Date.parse(b.updated_at || b.created_at || '') || 0
      return tb - ta
    })[0]

    // Only return if the newest job is still active
    return ACTIVE_STATUSES.has(newest.status) ? newest : null
  }, [jobs])

  return {
    jobs,
    loading,
    error,
    createJob,
    refreshJobs,
    pauseJob,
    resumeJob,
    stopJob,
    cancelJob,
    getJobForProject,
    setJobs,
    patchJobLive,
  }
}
