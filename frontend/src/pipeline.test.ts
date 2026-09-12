/**
 * Tests unitarios del state machine del pipeline.
 *
 * Cubre los escenarios que rompían la UI:
 * - Polling/SSE race condition (snapshot stale con current_step avanzado pero
 *   stepsCompleted incompleto).
 * - Transición active → done donde nunca llegó el evento 100% intermedio.
 * - Restart desde un step intermedio.
 * - Pipeline completado.
 *
 * Ejecutar: `npm run test` (o `npx vitest run`).
 */
import { describe, it, expect } from 'vitest'
import {
  getOverallPipelineProgress,
  getPipelineStepPercent,
  getPipelineStepState,
  MAIN_PIPELINE_STEPS,
  normalizePipelineStep,
} from './pipeline'

type Snapshot = Parameters<typeof getPipelineStepState>[1]

const baseSnapshot = (override: Partial<Snapshot> = {}): Snapshot => ({
  status: 'running',
  steps: [...MAIN_PIPELINE_STEPS],
  currentStep: null,
  progress: 0,
  stepsCompleted: [],
  elapsedSeconds: 0,
  startedAt: new Date().toISOString(),
  stepDurations: {},
  ...override,
})

describe('normalizePipelineStep — aliases legacy', () => {
  it('detect → face_track', () => {
    expect(normalizePipelineStep('detect')).toBe('face_positions')
  })
  it('guion → analyze', () => {
    expect(normalizePipelineStep('guion')).toBe('analyze')
  })
  it('ranking → rank', () => {
    expect(normalizePipelineStep('ranking')).toBe('rank')
  })
  it('step desconocido → null', () => {
    expect(normalizePipelineStep('xxxx')).toBe(null)
  })
  it('preserva steps v3', () => {
    expect(normalizePipelineStep('face_positions')).toBe('face_positions')
    expect(normalizePipelineStep('analyze')).toBe('analyze')
  })
})

describe('getPipelineStepState', () => {
  it('step en stepsCompleted → done', () => {
    const snap = baseSnapshot({ stepsCompleted: ['download', 'transcribe'] })
    expect(getPipelineStepState('download', snap)).toBe('done')
    expect(getPipelineStepState('transcribe', snap)).toBe('done')
  })

  it('step es currentStep → active', () => {
    const snap = baseSnapshot({ currentStep: 'diarize', stepsCompleted: ['download', 'transcribe'] })
    expect(getPipelineStepState('diarize', snap)).toBe('active')
  })

  it('step posterior → pending', () => {
    const snap = baseSnapshot({ currentStep: 'diarize', stepsCompleted: ['download', 'transcribe'] })
    expect(getPipelineStepState('speaker_bind', snap)).toBe('pending')
  })

  it('step antes del primer planned (reanudado) → done implícito', () => {
    // Escenario: user re-lanza desde "face_track" — los previos se asumen completados.
    const snap = baseSnapshot({
      steps: ['face_positions', 'speaker_bind', 'analyze', 'edit', 'rank'],
      currentStep: 'face_positions',
      stepsCompleted: [],
    })
    expect(getPipelineStepState('download', snap)).toBe('done')
    expect(getPipelineStepState('transcribe', snap)).toBe('done')
    expect(getPipelineStepState('diarize', snap)).toBe('done')
    expect(getPipelineStepState('face_positions', snap)).toBe('active')
    expect(getPipelineStepState('speaker_bind', snap)).toBe('pending')
  })
})

describe('getPipelineStepPercent', () => {
  it('done → 100 siempre', () => {
    const snap = baseSnapshot({
      stepsCompleted: ['download', 'transcribe', 'face_positions'],
      currentStep: 'speaker_bind',
      progress: 12,
    })
    expect(getPipelineStepPercent('face_positions', snap)).toBe(100)
    expect(getPipelineStepPercent('transcribe', snap)).toBe(100)
  })

  it('active → clamp del snapshot.progress', () => {
    const snap = baseSnapshot({ currentStep: 'transcribe', progress: 42 })
    expect(getPipelineStepPercent('transcribe', snap)).toBe(42)
  })

  it('pending → 0', () => {
    const snap = baseSnapshot({ currentStep: 'transcribe' })
    expect(getPipelineStepPercent('rank', snap)).toBe(0)
  })

  it('progress null → 0 (active sin señal)', () => {
    const snap = baseSnapshot({ currentStep: 'transcribe', progress: null })
    expect(getPipelineStepPercent('transcribe', snap)).toBe(0)
  })
})

describe('getOverallPipelineProgress', () => {
  it('status=completed → 100', () => {
    expect(getOverallPipelineProgress(baseSnapshot({ status: 'completed' }))).toBe(100)
  })

  it('excluye export del total mientras no esté activo ni done', () => {
    // Pipeline v3.5 son 9 steps. Si todos los primeros 8 están done y export pending,
    // overallProgress debe marcar 100 (export es manual).
    const done = MAIN_PIPELINE_STEPS.filter(s => s !== 'export')
    const snap = baseSnapshot({
      status: 'awaiting_export',
      stepsCompleted: done,
      currentStep: null,
      progress: 0,
    })
    expect(getOverallPipelineProgress(snap)).toBe(100)
  })

  it('pipeline a medio: 3 done + 1 active@50% de 8 relevantes ≈ 44%', () => {
    // Pipeline v3.5: 9 steps totales, 8 efectivos (excluye export hasta activarse).
    const snap = baseSnapshot({
      stepsCompleted: ['download', 'transcribe', 'diarize'],
      currentStep: 'face_positions',
      progress: 50,
    })
    // 3 done + 0.5 activo = 3.5 / 8 ≈ 43.75 → 44
    const pct = getOverallPipelineProgress(snap)
    expect(pct).toBeGreaterThanOrEqual(42)
    expect(pct).toBeLessThanOrEqual(46)
  })
})

/**
 * Caso crítico: race condition polling/SSE.
 * El backend emite el último "face_track pct=100", luego transiciona a "asd pct=0".
 * Si el polling trae el snapshot {current_step:asd, progress:0, stepsCompleted:[...face_track]}
 * pero el SSE trajo antes {current_step:face_track, progress:55}, el frontend mergeaba ambos
 * y dejaba stepsCompleted desactualizado. Verificamos que con snapshot correcto la
 * función retorna el state que el pct robusto espera.
 */
describe('Race condition SSE/polling → state consistency', () => {
  it('snapshot con current_step=asd y face_track en completed → face_track es done', () => {
    const snap = baseSnapshot({
      currentStep: 'speaker_bind',
      progress: 0,
      stepsCompleted: ['download', 'transcribe', 'diarize', 'face_positions'],
    })
    expect(getPipelineStepState('face_positions', snap)).toBe('done')
    expect(getPipelineStepPercent('face_positions', snap)).toBe(100)
  })

  it('snapshot transitorio: current_step=asd pero stepsCompleted incompleto → face_track pending', () => {
    // Este es el snapshot buggy. La función core reportará "pending" porque
    // face_track NO está en completed ni es current. El fix en ProcessingStatus
    // es que stablePct ignora este caso mostrando el último valor conocido.
    const snap = baseSnapshot({
      currentStep: 'speaker_bind',
      progress: 0,
      stepsCompleted: ['download', 'transcribe', 'diarize'], // falta face_track
    })
    expect(getPipelineStepState('face_positions', snap)).toBe('pending')
    // El pct del core devuelve 0 — el fix anti-flicker (en el componente)
    // preserva el último valor conocido en el ref, no este 0.
    expect(getPipelineStepPercent('face_positions', snap)).toBe(0)
  })
})
