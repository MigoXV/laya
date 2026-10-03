import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import type { DecisionRequest, DecisionResponse, ModelInfo, QuestionType } from './types'

const modes: { type: QuestionType; label: string; explanation: string }[] = [
  { type: 'choice', label: '选择', explanation: '从候选项中选择最符合文本的一项。' },
  { type: 'score', label: '评分', explanation: '按从低到高的等级，计算期望评分。' },
  { type: 'noul', label: '真假', explanation: '判断一个命题成立的概率，不生成文本。' },
]
const presets: Record<QuestionType, { state: string; instructions: string; options: string }> = {
  choice: {
    state: '小李负责测试，小王只负责发布。',
    instructions: '谁负责测试？', options: '小李\n小王',
  },
  score: {
    state: '线上服务出现故障，所有用户都无法登录，需要立即处理。',
    instructions: '这个问题有多紧急？', options: '不紧急\n需要尽快处理\n需要立即处理',
  },
  noul: {
    state: '小李负责测试，小王只负责发布。',
    instructions: '小李负责测试吗？', options: '',
  },
}

class ApiError extends Error {
  constructor(public status: number, public code: string) { super(code) }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { ...init, signal: AbortSignal.timeout(65000) })
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(response.status, body?.error || `http_${response.status}`)
  if (!body) throw new Error('invalid_service_response')
  return body as T
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const messages: Record<string, string> = {
      service_unavailable: '无法连接推理服务。请启动服务，然后重新连接。',
      service_timeout: '等待服务响应超时。请检查服务状态后重试。',
      not_ready: '模型尚未就绪，请稍后重新连接。',
      queue_full: '服务队列已满，请稍后重试。',
      inflight_full: '服务正在处理较多请求，请稍后重试。',
      body_too_large: '输入过大，请缩短文本后重试。',
      state_token_budget_exceeded: '文本超过模型的 Token 上限，请缩短文本。',
      question_head_exceeded: '问题或选项过长，请缩短后重试。',
      invalid_request: '请求格式不符合服务要求，请检查问题和选项。',
      deadline_exceeded: '推理超过服务的时间限制，请稍后重试。',
    }
    return messages[error.code] || `服务请求失败（${error.status} · ${error.code}），请检查输入或重新连接。`
  }
  return '请求未完成，请检查网络与服务状态后重试。'
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`
const ms = (value: number) => `${value.toFixed(1)} ms`

export default function App() {
  const [type, setType] = useState<QuestionType>('choice')
  const [state, setState] = useState(presets.choice.state)
  const [instructions, setInstructions] = useState(presets.choice.instructions)
  const [options, setOptions] = useState(presets.choice.options)
  const [connection, setConnection] = useState<'checking' | 'ready' | 'offline'>('checking')
  const [connectionError, setConnectionError] = useState('')
  const [serviceUrl, setServiceUrl] = useState('')
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<{
    request: DecisionRequest; response: DecisionResponse; roundtrip: number
  } | null>(null)

  const criteria = options.split('\n').map((line) => line.trim()).filter(Boolean)
  const request: DecisionRequest = {
    state,
    questions: { q1: { type, instructions, ...(type === 'noul' ? {} : { criteria }) } },
  }
  const stale = result !== null && JSON.stringify(result.request) !== JSON.stringify(request)
  const mode = modes.find((item) => item.type === type)!

  const connect = useCallback(async () => {
    setConnection('checking')
    setConnectionError('')
    setInfo(null)
    try {
      const [config, readiness, metadata] = await Promise.all([
        api<{ service_url: string }>('/connection'),
        api<{ ready: boolean }>('/health/ready'),
        api<ModelInfo>('/v1/info'),
      ])
      setServiceUrl(config.service_url)
      if (!readiness.ready || !metadata.ready) throw new ApiError(503, 'not_ready')
      setInfo(metadata)
      setConnection('ready')
    } catch (failure) {
      setConnection('offline')
      setConnectionError(errorMessage(failure))
    }
  }, [])

  useEffect(() => { void connect() }, [connect])

  function loadPreset(next: QuestionType) {
    setType(next)
    setState(presets[next].state)
    setInstructions(presets[next].instructions)
    setOptions(presets[next].options)
    setError('')
  }

  async function run(event: FormEvent) {
    event.preventDefault()
    if (busy || connection !== 'ready') return
    if (!state.trim() || !instructions.trim()) {
      setError('请输入待判断文本和问题描述。')
      return
    }
    if (type !== 'noul' && (criteria.length < 2 || criteria.length > 16)) {
      setError('请输入 2–16 个选项或等级，每行一项。')
      return
    }
    if (type === 'choice' && new Set(criteria).size !== criteria.length) {
      setError('候选项不能重复，请修改后重试。')
      return
    }
    setBusy(true)
    setError('')
    const started = performance.now()
    try {
      const response = await api<DecisionResponse>('/v1/decisions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request),
      })
      if (!response.answers?.q1) throw new Error('invalid_service_response')
      setResult({ request, response, roundtrip: performance.now() - started })
    } catch (failure) {
      setError(errorMessage(failure))
      if (failure instanceof ApiError && [502, 503].includes(failure.status)) void connect()
    } finally {
      setBusy(false)
    }
  }

  const answer = result?.response.answers.q1
  const resultCriteria = result?.request.questions.q1.criteria || []
  const distribution = answer ? Object.entries(answer.probabilities).map(([key, probability]) => ({
    key, probability,
    label: answer.type === 'choice' ? key : answer.type === 'noul'
      ? (key === '1' ? '成立' : '不成立') : `${key} · ${resultCriteria[Number(key)]}`,
    selected: answer.type === 'choice' && key === answer.choice,
  })) : []

  return <>
    <a className="skip-link" href="#input">跳至输入区</a>
    <header className="topbar">
      <a className="brand" href="/" aria-label="Laya 工作台首页"><span className="brand-mark" aria-hidden="true">L</span><strong>laya</strong><span className="brand-divider" /><span>模型实验</span></a>
      <div className="connection-tools">
        <span className={`connection ${connection}`} role="status"><span className="status-dot" aria-hidden="true" />{connection === 'ready' ? '服务已就绪' : connection === 'checking' ? '正在连接' : '服务未连接'}</span>
        <button className="quiet compact" onClick={() => void connect()} disabled={busy || connection === 'checking'}>重新连接</button>
      </div>
    </header>

    <main>
      <div className="page-heading"><div><p className="eyebrow">Demo 01 / Typed decisions</p><h1>推理测试工作台</h1><p className="intro">给出文本与问题，观察模型如何选择、评分和判断。</p></div><p className="heading-note">本地模型 · 真实服务响应</p></div>
      {connectionError && <div className="notice" role="alert">{connectionError}</div>}

      <div className="workspace">
        <section className="input-panel" id="input" aria-labelledby="input-title">
          <div className="section-heading"><h2 id="input-title"><span className="step">01</span>输入与问题</h2><span className="subtle">修改内容，重新测试</span></div>
          <form onSubmit={(event) => void run(event)}>
            <fieldset disabled={busy}>
              <legend className="sr-only">推理输入</legend>
              <div className="field"><label htmlFor="state">待判断文本</label><textarea id="state" className="state-input" value={state} onChange={(event) => setState(event.target.value)} required /><span className="field-hint">描述事实或场景。模型只依据你提供的文本判断。</span></div>
              <div className="field"><span id="mode-label" className="field-label">问题类型</span><div className="mode-switch" role="group" aria-labelledby="mode-label">{modes.map((item) => <button key={item.type} type="button" aria-pressed={item.type === type} onClick={() => { setType(item.type); setOptions(presets[item.type].options); setError('') }}>{item.label}<span>{item.type}</span></button>)}</div><span className="field-hint">{mode.explanation}</span></div>
              <div className="field"><label htmlFor="instructions">问题描述</label><textarea id="instructions" className="question-input" maxLength={8192} value={instructions} onChange={(event) => setInstructions(event.target.value)} required /></div>
              {type !== 'noul' && <div className="field"><label htmlFor="options">{type === 'choice' ? '候选项' : '评分等级'}</label><textarea id="options" className="options-input" value={options} onChange={(event) => setOptions(event.target.value)} required aria-describedby="options-hint" /><span id="options-hint" className="field-hint">{type === 'choice' ? '每行一个候选项，2–16 项，不可重复。' : '每行一个等级，从低到高排列。等级从 0 开始。'}</span></div>}
              <div className="presets"><label htmlFor="preset">快速示例</label><select id="preset" value="" onChange={(event) => loadPreset(event.target.value as QuestionType)}><option value="" disabled>载入一个示例</option><option value="choice">选择 · 项目分工</option><option value="score">评分 · 故障紧急程度</option><option value="noul">真假 · 职责判断</option></select></div>
            </fieldset>
            {error && <p className="notice" role="alert" id="run-error">{error}</p>}
            <div className="run-toolbar"><button className="primary" type="submit" disabled={busy || connection !== 'ready'} aria-describedby={error ? 'run-error' : undefined}><span aria-hidden="true">{busy ? '◌' : '↗'}</span>{busy ? '正在推理…' : '运行推理'}</button><span className="subtle" role="status">{busy ? '等待模型返回结果' : connection !== 'ready' ? '连接服务后即可运行' : '每次运行一个问题'}</span></div>
          </form>
        </section>

        <section className="result-panel" aria-labelledby="result-title" aria-busy={busy}>
          <div className="section-heading"><h2 id="result-title"><span className="step">02</span>推理结果</h2><span className="subtle" role="status">{busy ? '正在推理' : result ? stale || error ? '上次成功结果' : '推理完成' : '等待首次运行'}</span></div>
          {!answer || !result ? <div className="empty-result"><svg width="56" height="56" viewBox="0 0 56 56" fill="none" aria-hidden="true"><path d="M12 13h32v30H12zM19 34V23m9 11V19m9 15v-8" stroke="currentColor" strokeWidth="1.5" /><path d="M19 38h18" stroke="currentColor" /></svg><h3>让判断变得可见</h3><p>运行一次推理后，这里会展示答案、<br />概率分布与实际耗时。</p><span className="empty-footnote">选择 / 评分 / 真假</span></div> : <>
            {stale && <p className="stale-note">输入已修改。下方为上次输入的结果，请重新运行。</p>}
            <div className="answer-summary"><p className="eyebrow">{answer.type === 'choice' ? '模型选择' : answer.type === 'score' ? '期望等级' : '命题成立概率'}</p><p className="answer-value" data-testid="answer-value">{answer.type === 'choice' ? answer.choice : answer.type === 'score' ? answer.score!.toFixed(3) : percent(answer.noul!)}</p><p className="subtle">{answer.type === 'score' ? `等级范围 0–${resultCriteria.length - 1}，显示概率加权的连续评分。` : result.request.questions.q1.instructions}</p></div>
            <div className="distribution"><div className="small-heading"><h3>概率分布</h3><span className="subtle">所有候选结果</span></div><ul aria-label="概率分布">{distribution.map((item) => <li key={item.key} className={item.selected ? 'winning' : ''} data-testid="probability-row"><div className="probability-label"><span>{item.label}{item.selected && <span className="chosen">已选择</span>}</span><strong data-testid="probability-value">{percent(item.probability)}</strong></div><div className="probability-track" aria-hidden="true"><div className="probability-fill" style={{ width: `${item.probability * 100}%` }} /></div></li>)}</ul></div>
            <div className="confidence"><span>分布集中度</span><strong>{percent(answer.confidence)}</strong><p>概率越集中，此值越高。它不是准确率，也不是经过校准的正确概率。</p></div>
            <dl className="timing-grid"><div><dt>服务端总耗时</dt><dd>{ms(result.response.timings.total_ms)}</dd></div><div><dt>浏览器往返耗时</dt><dd>{ms(result.roundtrip)}</dd></div><div><dt>模型推理</dt><dd>{ms(result.response.timings.inference_ms)}</dd></div><div><dt>输入 Token</dt><dd>{result.response.usage.input_tokens}</dd></div></dl>
            <details className="raw-data"><summary>查看请求与响应 JSON</summary><h3>本次请求</h3><pre data-testid="request-json">{JSON.stringify(result.request, null, 2)}</pre><h3>服务响应</h3><pre data-testid="response-json">{JSON.stringify(result.response, null, 2)}</pre></details>
          </>}
        </section>
      </div>

      <footer className="model-footer"><div><span className="footer-label">当前模型</span><strong>{info?.model || '未连接'}</strong>{info && <span>{info.device} / {info.dtype} / {info.runner}</span>}</div><div>{info && <span>Torch {info.torch_version} · 上限 {info.max_len} Token</span>}<span className="service-url">{serviceUrl}</span></div></footer>
    </main>
  </>
}
