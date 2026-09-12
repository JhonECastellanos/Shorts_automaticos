import { useState, useRef, useEffect, useLayoutEffect, useMemo } from 'react'
import type { CSSProperties } from 'react'
import type { ProgressEvent } from '../types'
import {
  formatPipelineDuration,
  getOverallPipelineProgress,
  getPipelineStepPercent,
  getPipelineStepState,
  getPipelineTiming,
  MAIN_PIPELINE_STEPS,
  normalizePipelineStep,
  PIPELINE_STEP_META,
  STEP_DESCRIPTIONS,
} from '../pipeline'
import './ProcessingStatus.css'

/**
 * Hook: interpola un número entero hacia un target usando requestAnimationFrame.
 * Útil para porcentajes que saltan (15 → 50) — el texto mostrado fluye suave.
 * - Duración base 450ms con easing cúbico (ease-out).
 * - Cuando el target es 100 (step completado) la animación se acelera a 250ms
 *   para dar sensación de "cerrar el paso" sin quedarse colgado.
 * - Si target cambia durante la animación, reinicia desde el valor actual.
 */
function SmoothPct({ value, stateClass }: { value: number; stateClass: 'done' | 'active' | 'pending' }) {
  // Duración agresiva cuando el destino es 100% — visualmente "completa rápido".
  const duration = value === 100 ? 280 : 450
  const v = useSmoothNumber(value, duration)
  return <span className={`step-node-pct step-pct-${stateClass}`}>{v}%</span>
}


export function useSmoothNumber(target: number, duration = 500): number {
  const [value, setValue] = useState(target)
  const startRef = useRef(target)
  const startTimeRef = useRef(0)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    startRef.current = value
    startTimeRef.current = performance.now()

    function step(now: number) {
      const t = Math.min(1, (now - startTimeRef.current) / duration)
      const eased = 1 - Math.pow(1 - t, 3)
      const next = startRef.current + (target - startRef.current) * eased
      setValue(Math.round(next))
      if (t < 1) rafRef.current = requestAnimationFrame(step)
    }
    rafRef.current = requestAnimationFrame(step)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, duration])

  return value
}

interface Props {
  jobId?: string
  status: string
  currentStep: string | null
  progress: number | null
  events: ProgressEvent[]
  stepsCompleted: string[]
  error: string | null
  steps: string[] | null
  startedAt: string | null
  elapsedSeconds: number | null
  stepDurations: Record<string, number>
  onPause?: () => void
  onResume?: () => void
  onStop?: () => void
  onCancel?: () => void
  exportProgress?: { current: number; total: number } | null
}

export default function ProcessingStatus({
  jobId: _jobId, status, currentStep, progress, events, stepsCompleted, error,
  steps, startedAt, elapsedSeconds, stepDurations, onPause, onResume, onStop, onCancel, exportProgress
}: Props) {
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null)
  const [nowMs, setNowMs] = useState(() => Date.now())
  const [tooltipShift, setTooltipShift] = useState(0)
  const [tooltipNudgeY, setTooltipNudgeY] = useState(0)
  const pipelineRef = useRef<HTMLDivElement>(null)
  const logEndRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement | null>(null)
  const stepRefs = useRef<Record<string, HTMLDivElement | null>>({})

  // Click outside listener
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (pipelineRef.current && !pipelineRef.current.contains(event.target as Node)) {
        setActiveTooltip(null)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    setNowMs(Date.now())
    if (!['running', 'pending'].includes(status)) return

    const id = window.setInterval(() => {
      setNowMs(Date.now())
    }, 1000)

    return () => window.clearInterval(id)
  }, [status, startedAt])

  // Scroll al último log cuando cambia
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  useLayoutEffect(() => {
    if (!activeTooltip || !tooltipRef.current) return
    const stepEl = stepRefs.current[activeTooltip]
    if (!stepEl) return

    const tt = tooltipRef.current.getBoundingClientRect()
    const stepRect = stepEl.getBoundingClientRect()

    const maxW = window.innerWidth
    let shift = 0
    if (stepRect.left + (stepRect.width / 2) + (tt.width / 2) > maxW - 20) {
      shift = -((stepRect.left + (stepRect.width / 2) + (tt.width / 2)) - (maxW - 20))
    }
    if (stepRect.left + (stepRect.width / 2) - (tt.width / 2) < 20) {
      shift = 20 - (stepRect.left + (stepRect.width / 2) - (tt.width / 2))
    }
    setTooltipShift(shift)

    // Ajuste vertical si se sale por arriba
    let nudgeY = 0
    if (stepRect.top - tt.height - 15 < 10) {
      nudgeY = 10 - (stepRect.top - tt.height - 15)
    }
    setTooltipNudgeY(nudgeY)
  }, [activeTooltip, events])

  const snapshot = useMemo(() => ({
    status,
    steps,
    currentStep,
    progress,
    stepsCompleted,
    elapsedSeconds,
    startedAt,
    stepDurations,
  }), [status, steps, currentStep, progress, stepsCompleted, elapsedSeconds, startedAt, stepDurations])

  const allSteps = MAIN_PIPELINE_STEPS
  const timing = getPipelineTiming(snapshot, nowMs)
  const rawOverallProgress = getOverallPipelineProgress(snapshot)
  const activeStep = timing.activeStep
  const capturedSecs = elapsedSeconds !== null ? Math.max(0, Math.round(elapsedSeconds)) : (timing.elapsedSecs > 0 ? timing.elapsedSecs : null)

  // ── Anti-flicker: estados y porcentajes solo avanzan, nunca retroceden ──
  const lastProgressRef = useRef(0)
  const stepStatesRef = useRef<Record<string, 'done' | 'active' | 'pending'>>({})
  const stepPercentsRef = useRef<Record<string, number>>({})
  const jobIdRef = useRef(_jobId)

  // Resetear refs cuando cambia el job
  if (jobIdRef.current !== _jobId) {
    jobIdRef.current = _jobId
    stepStatesRef.current = {}
    stepPercentsRef.current = {}
    lastProgressRef.current = 0
  }

  const STATE_RANK = { pending: 0, active: 1, done: 2 } as const

  // Inferencia extra: si un step posterior ya está `active`, todos los anteriores
  // deben estar `done` (aunque el polling del backend aún no haya traído
  // stepsCompleted actualizado). Esto cierra la ventana de 3s entre polls.
  const activeIdxInMain = (() => {
    const activeRaw = normalizePipelineStep(currentStep) ?? timing.activeStep
    if (!activeRaw) return -1
    return MAIN_PIPELINE_STEPS.indexOf(activeRaw)
  })()

  function stableState(step: string, raw: 'done' | 'active' | 'pending'): 'done' | 'active' | 'pending' {
    const prev = stepStatesRef.current[step]
    const stepIdx = MAIN_PIPELINE_STEPS.indexOf(step as never)

    // Override: si hay un step posterior activo, este step DEBE estar done.
    let effective: 'done' | 'active' | 'pending' = raw
    if (activeIdxInMain > stepIdx && stepIdx >= 0) {
      effective = 'done'
    }

    if (prev && STATE_RANK[effective] < STATE_RANK[prev]) return prev
    stepStatesRef.current[step] = effective
    return effective
  }

  // Último pct visto en SSE por step. Si el polling dice 0 pero SSE ya emitió
  // pct más alto para este step, usamos el pct del SSE como fallback.
  const sseLatestPct: Record<string, number> = useMemo(() => {
    const out: Record<string, number> = {}
    for (const ev of events) {
      if (ev.percent == null) continue
      const norm = normalizePipelineStep(ev.step)
      if (!norm) continue
      const prev = out[norm] ?? 0
      if (ev.percent > prev) out[norm] = ev.percent
    }
    return out
  }, [events])

  function stablePct(step: string, raw: number, state: 'done' | 'active' | 'pending'): number {
    // Un step completado SIEMPRE marca 100.
    if (state === 'done') {
      stepPercentsRef.current[step] = 100
      return 100
    }
    if (state === 'pending') {
      // No tocar ref — mantiene el último valor en caso de restart.
      return 0
    }
    const prev = stepPercentsRef.current[step] ?? 0
    // Fallback SSE: si el polling no ha traído pct nuevo pero SSE sí.
    const sse = sseLatestPct[step] ?? 0
    const effectiveRaw = Math.max(raw, sse)
    if (effectiveRaw >= prev || effectiveRaw < prev - 20) {
      stepPercentsRef.current[step] = effectiveRaw
      return effectiveRaw
    }
    return prev
  }

  const overallProgressRaw = (() => {
    const raw = rawOverallProgress
    const last = lastProgressRef.current
    if (raw >= last) { lastProgressRef.current = raw; return raw }
    if (raw < last - 20) { lastProgressRef.current = raw; return raw } // reset real
    return last
  })()
  const overallProgress = useSmoothNumber(overallProgressRaw, 600)

  const activeEvents = activeTooltip
    ? events.filter(ev => normalizePipelineStep(ev.step) === activeTooltip)
    : []
  const tooltipStyle = {
    '--tooltip-shift': `${tooltipShift}px`,
    '--tooltip-nudge-y': `${tooltipNudgeY}px`,
  } as CSSProperties

  // Determinar label del progreso — no mostrar "exportación" si export no está activo
  const isExporting = activeStep === 'export' && ['running', 'pending'].includes(status)
  const progressLabel = exportProgress
    ? `Exportando ${exportProgress.current}/${exportProgress.total} shorts`
    : isExporting
      ? 'Exportación en curso'
      : status === 'awaiting_export'
        ? 'Ranking completado — listo para exportar'
        : activeStep ? `${PIPELINE_STEP_META[activeStep].label} en curso`
        : ['completed'].includes(status) ? 'Pipeline completado'
        : ['failed', 'cancelled'].includes(status) ? 'Pipeline detenido'
        : 'Pipeline listo'

  return (
    <div className="pipeline-bar-wrap" ref={pipelineRef}>
      <div className="pipeline-progress-meta">
        <span className="pipeline-progress-label">
          {progressLabel}
        </span>
        <span className="pipeline-progress-value">{overallProgress}%</span>
      </div>

      {/* Barra de progreso global — width viene del valor RAW (sin smooth) para
           que CSS transition width:0.8s sea el único que anima. El texto del
           porcentaje sí usa smooth JS (useSmoothNumber). */}
      <div className="pipeline-progress">
        <div className="pipeline-progress-fill" style={{ width: `${Math.min(overallProgressRaw, 100)}%` }} />
      </div>

      {/* Stepper horizontal */}
      <div className="pipeline-stepper">
        {allSteps.map((step, idx) => {
          const state = stableState(step, getPipelineStepState(step, snapshot))
          const stepPercent = stablePct(step, getPipelineStepPercent(step, snapshot), state)
          const { label, icon } = PIPELINE_STEP_META[step]
          const isLast = idx === allSteps.length - 1
          return (
            <div
              key={step}
              className={`step-wrap ${activeTooltip === step ? 'tooltip-active' : ''}`}
              ref={node => { stepRefs.current[step] = node }}
              onClick={() => setActiveTooltip(prev => prev === step ? null : step)}
            >
              <div className={`step-node step-${state}`}>
                <span className="step-node-icon">
                  {state === 'done' ? '✓' : icon}
                </span>
              </div>
              <div className="step-node-copy">
                <span className={`step-node-label step-label-${state}`}>{label}</span>
                <SmoothPct value={stepPercent} stateClass={state} />
              </div>
              {!isLast && <div className={`step-connector ${state === 'done' ? 'connector-done' : ''}`} />}

              {/* Tooltip flotante — toggle on click */}
              {activeTooltip === step && (
                <div ref={tooltipRef} className="step-tooltip" style={tooltipStyle} onClick={e => e.stopPropagation()}>
                  <div className="step-tooltip-header">
                    <span className="step-tooltip-title">{icon} {label}</span>
                    <span className={`step-tooltip-state tag-${state}`}>
                      {state === 'done' ? 'Completado' : state === 'active' ? 'En progreso' : 'Pendiente'}
                    </span>
                    <span className="step-tooltip-pct">{stepPercent}%</span>
                    <span className="step-tooltip-est">⏱ {PIPELINE_STEP_META[step].est}</span>
                  </div>
                  
                  {STEP_DESCRIPTIONS[step] && (
                    <div className="step-tooltip-desc">{STEP_DESCRIPTIONS[step]}</div>
                  )}

                  <div className="step-tooltip-logs">
                    {activeEvents.length === 0 ? (
                      <span className="step-tooltip-empty">Sin actividad</span>
                    ) : (
                      activeEvents.map((ev, i) => (
                        <div key={i} className="tooltip-log-line">
                          <span>{ev.message}</span>
                          {ev.percent !== null && <span className="tooltip-log-pct">{ev.percent.toFixed(0)}%</span>}
                        </div>
                      ))
                    )}
                    <div ref={logEndRef} />
                  </div>
                  {error && state === 'active' && (
                    <div className="step-tooltip-error">{error}</div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Fila inferior: controles (izq) + cronómetro (der) */}
      <div className="pipeline-footer">
        <div className="pipeline-footer-left">
          <button className="pipeline-control" disabled={!onPause} onClick={onPause} type="button">
            ⏸ Pausar
          </button>
          <button className="pipeline-control pipeline-control-primary" disabled={!onResume} onClick={onResume} type="button">
            ▶ Play
          </button>
          <button className="pipeline-control pipeline-control-danger" disabled={!onStop} onClick={onStop} type="button">
            ⏹ Stop
          </button>
          <button className="pipeline-control pipeline-control-warning" disabled={!onCancel} onClick={onCancel} type="button">
            ✕ Cancelar
          </button>
          {error && <span className="pipeline-error-inline">{error}</span>}
        </div>
        {['running', 'pending'].includes(status) && (
          <div className="pipeline-timer">
            <span className="timer-total">{overallProgress}% total</span>
            <span className="timer-countdown">⌛ {formatPipelineDuration(timing.remainingSecs)}</span>
            <span className="timer-caption">restantes aprox.</span>
            <span className="timer-elapsed">transcurrido {formatPipelineDuration(timing.elapsedSecs)}</span>
            {timing.remainingSecs > 0 && (
              <span className="timer-remaining">ETA dinámica {formatPipelineDuration(timing.estimatedTotalSecs)}</span>
            )}
          </div>
        )}
        {status === 'completed' && capturedSecs !== null && (
          <div className="pipeline-timer">
            <span className="timer-total">100% total</span>
            <span className="timer-done">✓ completado en {formatPipelineDuration(capturedSecs)}</span>
            <span className="timer-frozen">referencia guardada</span>
          </div>
        )}
        {status === 'awaiting_export' && capturedSecs !== null && (
          <div className="pipeline-timer">
            <span className="timer-total">{overallProgress}% total</span>
            <span className="timer-stopped">shorts listos en {formatPipelineDuration(capturedSecs)}</span>
            <span className="timer-frozen">▶ Exportar desde la lista</span>
          </div>
        )}
        {(status === 'paused' || status === 'stopped') && capturedSecs !== null && (
          <div className="pipeline-timer">
            <span className="timer-total">{overallProgress}% total</span>
            <span className="timer-stopped">pausado en {formatPipelineDuration(capturedSecs)}</span>
            <span className="timer-frozen">listo para continuar</span>
          </div>
        )}
        {(status === 'failed' || status === 'cancelled') && capturedSecs !== null && (
          <div className="pipeline-timer">
            <span className="timer-total">{overallProgress}% total</span>
            <span className="timer-stopped">detenido en {formatPipelineDuration(capturedSecs)}</span>
            <span className="timer-frozen">referencia parcial guardada</span>
          </div>
        )}
      </div>
    </div>
  )
}
