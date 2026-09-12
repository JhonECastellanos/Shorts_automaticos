/**
 * Harness interno para validar transiciones de ProcessingStatus sin backend.
 *
 * Cómo usar: importar desde App temporalmente:
 *   import TestHarness from './components/__ProcessingStatusHarness'
 *   return <TestHarness />
 *
 * Simula secuencias de props para verificar que:
 * - El pct del step activo sube fluido (0 → 25 → 60 → 100)
 * - Al pasar a done, el icono transiciona a ✓ con pop animation
 * - Cuando step N+1 está active, los pasos 0..N se fuerzan a done (sin esperar polling)
 * - SSE events pueden adelantarse al polling (fallback pct desde SSE)
 * - Reiniciar desde un paso intermedio resetea refs correctamente
 */
import { useState, useEffect } from 'react'
import ProcessingStatus from './ProcessingStatus'
import { MAIN_PIPELINE_STEPS } from '../pipeline'
import type { ProgressEvent } from '../types'

type Phase = {
  name: string
  props: {
    status: string
    currentStep: string | null
    progress: number | null
    stepsCompleted: string[]
    events: ProgressEvent[]
  }
  delayMs: number
}

const jid = 'test-harness'

function ev(step: string, message: string, percent: number | null): ProgressEvent {
  return { job_id: jid, step, message, percent }
}

const sequence: Phase[] = [
  // 1. diarize arrancando
  {
    name: '1. diarize 0%',
    delayMs: 0,
    props: {
      status: 'running', currentStep: 'diarize', progress: 0, stepsCompleted: [],
      events: [ev('diarize', 'Iniciando diarize...', 0)],
    },
  },
  // 2. diarize subiendo
  {
    name: '2. diarize 50%',
    delayMs: 2000,
    props: {
      status: 'running', currentStep: 'diarize', progress: 50, stepsCompleted: [],
      events: [
        ev('diarize', 'Iniciando diarize...', 0),
        ev('diarize', 'Extrayendo MFCC...', 35),
        ev('diarize', '6730 ventanas de voz', 50),
      ],
    },
  },
  // 3. SSE adelanta con 80% antes que polling (simula lag de polling)
  {
    name: '3. SSE 80% pero polling aún en 50',
    delayMs: 2000,
    props: {
      status: 'running', currentStep: 'diarize', progress: 50, stepsCompleted: [],
      events: [
        ev('diarize', 'Iniciando diarize...', 0),
        ev('diarize', 'Extrayendo MFCC...', 35),
        ev('diarize', '6730 ventanas de voz', 50),
        ev('diarize', 'Smoothing temporal...', 80),
      ],
    },
  },
  // 4. Transición a face_track — diarize debería ir a 100 + icono ✓
  {
    name: '4. face_track active (diarize debería ir a done aunque polling lento)',
    delayMs: 2500,
    props: {
      status: 'running', currentStep: 'face_track', progress: 20, stepsCompleted: ['diarize'],
      events: [
        ev('diarize', 'Diarización completada', 100),
        ev('face_track', 'Iniciando face_track...', 0),
        ev('face_track', 'Muestra 30/120', 20),
      ],
    },
  },
  // 5. face_track avanza
  {
    name: '5. face_track 70%',
    delayMs: 2000,
    props: {
      status: 'running', currentStep: 'face_track', progress: 70, stepsCompleted: ['diarize'],
      events: [
        ev('face_track', 'Muestra 84/120', 70),
      ],
    },
  },
  // 6. Caso problemático: ASD arrancó pero polling aún dice face_track=70
  //    (simula lag). Override: face_track debe forzarse a done cuando SSE
  //    reporta asd activo.
  {
    name: '6. ASD ya corriendo en SSE, polling aún no actualizó',
    delayMs: 2000,
    props: {
      // polling aún tiene face_track como currentStep
      status: 'running', currentStep: 'face_track', progress: 70, stepsCompleted: ['diarize'],
      events: [
        ev('face_track', 'Muestra 120/120 — completado', 100),
        ev('face_track', '5 tracks generados', 100),
        ev('asd', 'Iniciando asd...', 0),
        ev('asd', 'Analizando 8414 ventanas', 10),
        ev('asd', 'Procesando ventana 500/8414', 20),
      ],
    },
  },
  // 7. Polling finalmente cataches up
  {
    name: '7. Polling catches up — ASD active, face_track done, pct del SSE usado',
    delayMs: 2000,
    props: {
      status: 'running', currentStep: 'asd', progress: 30, stepsCompleted: ['diarize', 'face_track'],
      events: [
        ev('asd', 'Procesando ventana 1500/8414', 30),
      ],
    },
  },
  // 8. Final: completed
  {
    name: '8. Todos completados',
    delayMs: 3000,
    props: {
      status: 'awaiting_export', currentStep: null, progress: 100,
      stepsCompleted: MAIN_PIPELINE_STEPS.slice(0, -1),
      events: [ev('rank', 'Ranking completado', 100)],
    },
  },
]

export default function ProcessingStatusHarness() {
  const [idx, setIdx] = useState(0)

  useEffect(() => {
    if (idx >= sequence.length - 1) return
    const t = setTimeout(() => setIdx(i => Math.min(i + 1, sequence.length - 1)), sequence[idx + 1].delayMs)
    return () => clearTimeout(t)
  }, [idx])

  const phase = sequence[idx]

  return (
    <div style={{ padding: 20, background: '#1a1a1a', minHeight: '100vh', color: '#eee' }}>
      <h2>ProcessingStatus Harness — Phase {idx + 1} / {sequence.length}</h2>
      <p style={{ color: '#aaa' }}>{phase.name}</p>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <button onClick={() => setIdx(0)}>Reset</button>
        <button onClick={() => setIdx(Math.max(0, idx - 1))}>Prev</button>
        <button onClick={() => setIdx(Math.min(sequence.length - 1, idx + 1))}>Next</button>
        {sequence.map((_, i) => (
          <button key={i} onClick={() => setIdx(i)} style={{ opacity: i === idx ? 1 : 0.5 }}>
            {i + 1}
          </button>
        ))}
      </div>

      <ProcessingStatus
        jobId={jid}
        status={phase.props.status}
        currentStep={phase.props.currentStep}
        progress={phase.props.progress}
        events={phase.props.events}
        stepsCompleted={phase.props.stepsCompleted}
        error={null}
        steps={MAIN_PIPELINE_STEPS}
        startedAt={new Date(Date.now() - 60_000).toISOString()}
        elapsedSeconds={60}
        stepDurations={{}}
      />

      <pre style={{ marginTop: 20, fontSize: 11, color: '#888' }}>
{JSON.stringify(phase.props, null, 2)}
      </pre>
    </div>
  )
}
