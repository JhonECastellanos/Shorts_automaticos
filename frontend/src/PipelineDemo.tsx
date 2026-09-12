/**
 * Demo visual del ProcessingStatus sin backend.
 *
 * Permite reproducir los escenarios problemáticos que reportó el usuario
 * y verificar que la UI los maneja correctamente:
 *
 * 1. **Flujo normal**: cada step sube 0→100 suave, cambia a done, siguiente arranca.
 * 2. **Gap de evento (bug reportado)**: face_track sube a 55% y **sin emitir 100%**
 *    pasa a `done`. La UI debe cerrar face_track a 100 y pasar a siguiente sin dejar
 *    el pct colgado.
 * 3. **Race SSE/polling**: current_step avanza pero stepsCompleted llega tarde.
 * 4. **Restart intermedio**: pipeline arranca desde face_track, los previos se
 *    marcan done implícitos.
 * 5. **Pipeline completado**: todos los steps done salvo export (awaiting_export).
 *
 * Uso: navegar a `http://localhost:5173/?demo=pipeline`
 */
import { useState, useRef, useEffect } from 'react'
import ProcessingStatus from './components/ProcessingStatus'
import { MAIN_PIPELINE_STEPS } from './pipeline'
import type { ProgressEvent } from './types'

type Snapshot = {
  status: string
  currentStep: string | null
  progress: number
  stepsCompleted: string[]
  elapsedSeconds: number
}

type Scenario = {
  name: string
  description: string
  frames: Array<{ wait: number; patch: Partial<Snapshot>; event?: ProgressEvent }>
}

const JOB_ID = 'demo'

// ── Escenarios ─────────────────────────────────────────────────────

const SCENARIOS: Scenario[] = [
  {
    name: 'Flujo normal (todos los steps)',
    description: 'Cada step sube 0→100 y pasa al siguiente. Caso ideal.',
    frames: buildNormalFlow(),
  },
  {
    name: 'Bug: face_track cierra sin emitir 100%',
    description:
      'face_track sube hasta 55%, backend avanza a asd y reporta face_track en stepsCompleted. '
      + 'La UI debe cerrar face_track a 100% sin dejar colgado el 55%.',
    frames: buildStalePctFlow(),
  },
  {
    name: 'Race SSE/polling (stepsCompleted stale)',
    description:
      'current_step avanza a asd con progress=0 pero stepsCompleted aún no incluye face_track. '
      + 'Después llega el update real. La UI no debe retroceder a pending.',
    frames: buildRaceConditionFlow(),
  },
  {
    name: 'Restart desde paso intermedio',
    description:
      'Pipeline reiniciado desde face_track. Los 3 primeros se consideran done implícito.',
    frames: buildRestartFlow(),
  },
  {
    name: 'Completado (awaiting_export)',
    description: 'Todos los steps done salvo export. Overall = 100%, export manual pendiente.',
    frames: buildAwaitingExportFlow(),
  },
]

function buildNormalFlow(): Scenario['frames'] {
  const frames: Scenario['frames'] = []
  const stepList = MAIN_PIPELINE_STEPS.filter(s => s !== 'export')
  for (const step of stepList) {
    // arranca el step
    frames.push({
      wait: 200,
      patch: { status: 'running', currentStep: step, progress: 0 },
      event: { job_id: JOB_ID, step, message: `Iniciando ${step}`, percent: 0 },
    })
    for (let p = 10; p <= 100; p += 10) {
      frames.push({
        wait: 150,
        patch: { currentStep: step, progress: p },
        event: { job_id: JOB_ID, step, message: `${step} ${p}%`, percent: p },
      })
    }
    // done
    frames.push({
      wait: 100,
      patch: ((prev: Snapshot) => ({
        stepsCompleted: [...prev.stepsCompleted, step],
      })) as unknown as Partial<Snapshot>,
    })
  }
  frames.push({
    wait: 200,
    patch: { status: 'awaiting_export', currentStep: null, progress: 0 },
  })
  return frames
}

function buildStalePctFlow(): Scenario['frames'] {
  const frames: Scenario['frames'] = []
  // download completa
  frames.push({ wait: 200, patch: { status: 'running', currentStep: 'download', progress: 0 } })
  frames.push({ wait: 200, patch: { currentStep: 'download', progress: 100 } })
  frames.push({ wait: 150, patch: ((prev: Snapshot) => ({ stepsCompleted: [...prev.stepsCompleted, 'download'] })) as unknown as Partial<Snapshot> })
  // transcribe completa
  frames.push({ wait: 200, patch: { currentStep: 'transcribe', progress: 100 } })
  frames.push({ wait: 150, patch: ((prev: Snapshot) => ({ stepsCompleted: [...prev.stepsCompleted, 'transcribe'] })) as unknown as Partial<Snapshot> })
  // diarize completa
  frames.push({ wait: 200, patch: { currentStep: 'diarize', progress: 100 } })
  frames.push({ wait: 150, patch: ((prev: Snapshot) => ({ stepsCompleted: [...prev.stepsCompleted, 'diarize'] })) as unknown as Partial<Snapshot> })
  // face_track sube a 55 y se salta el 100 → pasa directo a asd
  frames.push({ wait: 200, patch: { currentStep: 'face_track', progress: 10 } })
  frames.push({ wait: 400, patch: { currentStep: 'face_track', progress: 35 } })
  frames.push({ wait: 400, patch: { currentStep: 'face_track', progress: 55 } })
  // el backend PASA directo a asd sin emitir 100 intermedio
  frames.push({
    wait: 600,
    patch: ((prev: Snapshot) => ({
      currentStep: 'asd',
      progress: 0,
      stepsCompleted: [...prev.stepsCompleted, 'face_track'],
    })) as unknown as Partial<Snapshot>,
  })
  // asd avanza
  frames.push({ wait: 300, patch: { currentStep: 'asd', progress: 50 } })
  frames.push({ wait: 300, patch: { currentStep: 'asd', progress: 100 } })
  frames.push({ wait: 150, patch: ((prev: Snapshot) => ({ stepsCompleted: [...prev.stepsCompleted, 'asd'] })) as unknown as Partial<Snapshot> })
  return frames
}

function buildRaceConditionFlow(): Scenario['frames'] {
  const frames: Scenario['frames'] = []
  // Hasta face_track a 80%
  frames.push({ wait: 200, patch: { status: 'running', currentStep: 'face_track', progress: 80, stepsCompleted: ['download', 'transcribe', 'diarize'] } })
  // Polling trae snapshot STALE: current_step=asd, progress=0, stepsCompleted NO incluye face_track
  frames.push({ wait: 400, patch: { currentStep: 'asd', progress: 0, stepsCompleted: ['download', 'transcribe', 'diarize'] } })
  // Siguiente poll trae el estado correcto
  frames.push({ wait: 500, patch: { currentStep: 'asd', progress: 0, stepsCompleted: ['download', 'transcribe', 'diarize', 'face_track'] } })
  frames.push({ wait: 400, patch: { currentStep: 'asd', progress: 100 } })
  frames.push({ wait: 150, patch: ((prev: Snapshot) => ({ stepsCompleted: [...prev.stepsCompleted, 'asd'] })) as unknown as Partial<Snapshot> })
  return frames
}

function buildRestartFlow(): Scenario['frames'] {
  const frames: Scenario['frames'] = []
  frames.push({
    wait: 200,
    patch: {
      status: 'running',
      currentStep: 'face_track',
      progress: 0,
      stepsCompleted: [],
    },
  })
  for (let p = 10; p <= 100; p += 10) {
    frames.push({ wait: 150, patch: { currentStep: 'face_track', progress: p } })
  }
  frames.push({ wait: 200, patch: ((prev: Snapshot) => ({ stepsCompleted: [...prev.stepsCompleted, 'face_track'] })) as unknown as Partial<Snapshot> })
  return frames
}

function buildAwaitingExportFlow(): Scenario['frames'] {
  const frames: Scenario['frames'] = []
  frames.push({
    wait: 100,
    patch: {
      status: 'awaiting_export',
      currentStep: null,
      progress: 0,
      stepsCompleted: MAIN_PIPELINE_STEPS.filter(s => s !== 'export'),
    },
  })
  return frames
}

// ── Componente ──────────────────────────────────────────────────────

export default function PipelineDemo() {
  const [scenarioIdx, setScenarioIdx] = useState(0)
  const [snapshot, setSnapshot] = useState<Snapshot>({
    status: 'pending',
    currentStep: null,
    progress: 0,
    stepsCompleted: [],
    elapsedSeconds: 0,
  })
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const cancelRef = useRef(false)
  const elapsedIntervalRef = useRef<number | null>(null)
  const jobIdRef = useRef(JOB_ID + '-' + Date.now())

  // Cronómetro para elapsedSeconds
  useEffect(() => {
    if (!playing) {
      if (elapsedIntervalRef.current !== null) {
        window.clearInterval(elapsedIntervalRef.current)
        elapsedIntervalRef.current = null
      }
      return
    }
    elapsedIntervalRef.current = window.setInterval(() => {
      setSnapshot(s => ({ ...s, elapsedSeconds: s.elapsedSeconds + 1 }))
    }, 1000)
    return () => {
      if (elapsedIntervalRef.current !== null) {
        window.clearInterval(elapsedIntervalRef.current)
        elapsedIntervalRef.current = null
      }
    }
  }, [playing])

  async function runScenario(idx: number) {
    cancelRef.current = false
    setPlaying(true)
    jobIdRef.current = `${JOB_ID}-${Date.now()}`
    setSnapshot({
      status: 'pending',
      currentStep: null,
      progress: 0,
      stepsCompleted: [],
      elapsedSeconds: 0,
    })
    setEvents([])

    await new Promise(r => setTimeout(r, 200 / speed))

    const scenario = SCENARIOS[idx]
    for (const frame of scenario.frames) {
      if (cancelRef.current) break
      await new Promise(r => setTimeout(r, frame.wait / speed))
      setSnapshot(prev => {
        // patch puede ser objeto estático o función
        const patch = typeof frame.patch === 'function'
          ? (frame.patch as unknown as (p: Snapshot) => Partial<Snapshot>)(prev)
          : frame.patch
        return { ...prev, ...patch }
      })
      if (frame.event) {
        setEvents(prev => [...prev, frame.event as ProgressEvent])
      }
    }
    setPlaying(false)
  }

  function stopScenario() {
    cancelRef.current = true
    setPlaying(false)
  }

  const currentScenario = SCENARIOS[scenarioIdx]

  return (
    <div style={{ padding: '24px', fontFamily: 'system-ui, sans-serif', background: 'var(--bg-primary)', minHeight: '100vh', color: 'var(--text-primary)' }}>
      <h1>Pipeline UI — Demo</h1>
      <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
        Simula secuencias de eventos sin backend para verificar comportamiento UX.
      </p>

      <div style={{ display: 'flex', gap: 12, marginTop: 16, marginBottom: 20, flexWrap: 'wrap', alignItems: 'center' }}>
        <label>
          Escenario:{' '}
          <select value={scenarioIdx} onChange={e => setScenarioIdx(Number(e.target.value))} disabled={playing} style={{ padding: '6px 10px' }}>
            {SCENARIOS.map((s, i) => <option key={i} value={i}>{s.name}</option>)}
          </select>
        </label>
        <label>
          Velocidad: {speed}x{' '}
          <input type="range" min="0.5" max="5" step="0.5" value={speed} onChange={e => setSpeed(Number(e.target.value))} style={{ verticalAlign: 'middle' }} />
        </label>
        <button onClick={() => runScenario(scenarioIdx)} disabled={playing} style={{ padding: '6px 14px' }}>
          ▶ Play
        </button>
        <button onClick={stopScenario} disabled={!playing} style={{ padding: '6px 14px' }}>
          ⏹ Stop
        </button>
      </div>

      <div style={{ background: 'var(--bg-secondary)', padding: 10, borderRadius: 6, marginBottom: 20, fontSize: 13 }}>
        <strong>{currentScenario.name}</strong>
        <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>{currentScenario.description}</div>
      </div>

      <div style={{ background: 'var(--bg-secondary)', padding: 16, borderRadius: 8, border: '1px solid var(--border)' }}>
        <ProcessingStatus
          jobId={jobIdRef.current}
          status={snapshot.status}
          currentStep={snapshot.currentStep}
          progress={snapshot.progress}
          events={events}
          stepsCompleted={snapshot.stepsCompleted}
          error={null}
          steps={MAIN_PIPELINE_STEPS as unknown as string[]}
          startedAt={new Date(Date.now() - snapshot.elapsedSeconds * 1000).toISOString()}
          elapsedSeconds={snapshot.elapsedSeconds}
          stepDurations={{}}
        />
      </div>

      <details style={{ marginTop: 20 }}>
        <summary style={{ cursor: 'pointer', color: 'var(--text-muted)' }}>Snapshot debug</summary>
        <pre style={{ background: 'var(--bg-secondary)', padding: 10, borderRadius: 6, fontSize: 11 }}>
{JSON.stringify(snapshot, null, 2)}
        </pre>
      </details>
    </div>
  )
}
