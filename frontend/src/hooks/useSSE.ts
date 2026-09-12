import { useEffect, useRef, useState, useCallback } from 'react'
import { api } from '../api'
import type { ProgressEvent } from '../types'

interface UseSSEOptions {
  /**
   * Callback llamado con cada evento nuevo (solo en vivo vía SSE, no los
   * históricos cargados al inicio). Útil para propagar el progreso al
   * state del job en tiempo real sin esperar al polling.
   */
  onLiveEvent?: (event: ProgressEvent) => void
}

export function useSSE(
  jobId: string | null,
  logsOnlyJobId?: string | null,
  options: UseSSEOptions = {},
) {
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [connected, setConnected] = useState(false)
  const sourceRef = useRef<EventSource | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const retryCountRef = useRef(0)
  const MAX_RETRIES = 5
  // Throttle buffer: collect events during 300ms window, then flush
  const bufferRef = useRef<ProgressEvent[]>([])
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Ref para acceder al callback más reciente sin re-suscribir el effect
  const onLiveEventRef = useRef(options.onLiveEvent)
  onLiveEventRef.current = options.onLiveEvent

  const flushBuffer = useCallback(() => {
    flushTimerRef.current = null
    setEvents(prev => {
      const buf = bufferRef.current
      if (buf.length === 0) return prev
      bufferRef.current = []
      return [...prev, ...buf]
    })
  }, [])

  // The effective jobId for loading logs: prefer the SSE jobId, fallback to logsOnlyJobId
  const effectiveLogId = jobId ?? logsOnlyJobId ?? null

  useEffect(() => {
    let cancelled = false

    if (!effectiveLogId) {
      setEvents([])
      sourceRef.current?.close()
      setConnected(false)
      return
    }

    retryCountRef.current = 0

    // Load historical logs first (for both live and non-live jobs)
    void api.getLogs(effectiveLogId)
      .then(({ log }) => {
        if (cancelled) return
        const historical = log.map(entry => ({
          job_id: effectiveLogId,
          step: entry.step,
          message: entry.msg,
          percent: entry.pct,
        }))
        setEvents(historical)
      })
      .catch(() => {
        if (!cancelled) setEvents([])
      })

    // Only connect SSE for live jobs (when jobId is provided)
    if (!jobId) return () => { cancelled = true }

    function connectSSE() {
      if (cancelled) return
      if (retryCountRef.current >= MAX_RETRIES) return

      const source = new EventSource(`/api/jobs/${jobId}/progress`)
      sourceRef.current = source

      source.onopen = () => {
        if (!cancelled) setConnected(true)
      }
      source.onmessage = (e) => {
        if (cancelled) return
        retryCountRef.current = 0 // Reset only when we receive real data
        try {
          const event: ProgressEvent = JSON.parse(e.data)
          // Dispatch live event SINCRONAMENTE al callback del consumer para
          // actualización instantánea de UI (barra, stepper, pct). El buffer
          // de 300ms es solo para el state `events` usado en tooltips/logs.
          try { onLiveEventRef.current?.(event) } catch { /* no-op */ }

          if (event.step === 'done') {
            // Flush immediately for terminal events
            if (flushTimerRef.current) { clearTimeout(flushTimerRef.current); flushTimerRef.current = null }
            const pending = [...bufferRef.current, event]
            bufferRef.current = []
            setEvents(prev => [...prev, ...pending])
            source.close()
            setConnected(false)
          } else {
            bufferRef.current.push(event)
            if (!flushTimerRef.current) {
              flushTimerRef.current = setTimeout(flushBuffer, 300)
            }
          }
        } catch { /* ignore parse errors */ }
      }
      source.onerror = () => {
        setConnected(false)
        source.close()
        if (!cancelled && retryCountRef.current < MAX_RETRIES) {
          retryCountRef.current++
          const delay = Math.min(2000 * Math.pow(2, retryCountRef.current - 1), 30000)
          retryRef.current = setTimeout(connectSSE, delay)
        }
      }
    }

    connectSSE()

    return () => {
      cancelled = true
      if (retryRef.current) clearTimeout(retryRef.current)
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current)
      bufferRef.current = []
      sourceRef.current?.close()
      setConnected(false)
    }
  }, [jobId, effectiveLogId, flushBuffer])

  const reset = useCallback(() => setEvents([]), [])

  return { events, connected, reset }
}
