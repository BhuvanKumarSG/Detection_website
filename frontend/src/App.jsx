import React, { useState, useRef } from 'react'

// Use Vite env var for backend URL (set VITE_API_URL in Netlify env)
const API = import.meta.env.VITE_API_URL ?? 'http://localhost:5000'

export default function App() {
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [useBase, setUseBase] = useState(true)
  const [useBio, setUseBio] = useState(false)
  const fileRef = useRef()

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  function onFileChange(e) {
    const f = e.target.files[0]
    if (f) setFile(f)
  }

  async function onSubmit(e) {
    e.preventDefault()
    if (!file) return alert('Please provide an audio file')
    const form = new FormData()
    form.append('file', file)
    if (useBase) form.append('models', 'base')
    if (useBio) form.append('models', 'bio')

    try {
      setLoading(true)
      setResults(null)
      const resp = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: form
      })
      const json = await resp.json()
      setResults(json)
    } catch (err) {
      setResults({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1>Audio Spoof Detection</h1>
      <form onSubmit={onSubmit} className="panel">
        <div
          className={`drop ${dragOver ? 'over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current && fileRef.current.click()}
        >
          {file ? <div className="file-name">{file.name}</div> : <div>Drag & drop an audio file here, or click to select</div>}
          <input ref={fileRef} type="file" accept="audio/*" onChange={onFileChange} style={{ display: 'none' }} />
        </div>

        <div className="controls">
          <label><input type="checkbox" checked={useBase} onChange={(e) => setUseBase(e.target.checked)} /> Use base model</label>
          <label><input type="checkbox" checked={useBio} onChange={(e) => setUseBio(e.target.checked)} /> Use bio model</label>
        </div>

        <div className="actions">
          <button type="submit" disabled={loading}>{loading ? 'Analyzing...' : 'Analyze'}</button>
        </div>
      </form>

      <div className="results">
        <h2>Results</h2>
        {results ? (
          <div className="results-grid">
            {Object.entries(results).map(([name, data]) => {
              const title = name === 'base' ? 'Base model' : name === 'bio' ? 'Bio model' : name
              const status = data && data.status_code ? data.status_code : data && data.status ? data.status : 'unknown'

              // prefer structured result if present
              let label = 'N/A'
              let confidence = null
              if (data && data.result && Array.isArray(data.result.predict) && Array.isArray(data.result.predict_proba)) {
                const pred = data.result.predict[0]
                const probs = data.result.predict_proba[0] || []
                // when probs length >= 2 assume [real, spoof]
                if (probs.length >= 2) {
                  const spoofProb = probs[1]
                  const realProb = probs[0]
                  label = pred === 1 || spoofProb > realProb ? 'Spoof' : 'Real'
                  confidence = Math.max(spoofProb, realProb)
                } else if (probs.length === 1) {
                  // single probability interpreted as confidence for positive class
                  label = pred === 1 ? 'Spoof' : 'Real'
                  confidence = probs[0]
                } else {
                  label = pred === 1 ? 'Spoof' : 'Real'
                }
              }

              return (
                <div className="model-card" key={name}>
                  <div className="model-card-header">
                    <div className="model-title">{title}</div>
                    <div className={`status ${status === 'local' ? 'local' : status === 200 || status === 'ok' ? 'ok' : 'unknown'}`}>{String(status)}</div>
                  </div>

                  {data && data.local_error ? (
                    <div className="error">{data.local_error}</div>
                  ) : null}

                  {label !== 'N/A' ? (
                    <div className="result-main">
                      <div className={`label ${label === 'Spoof' ? 'spoof' : 'real'}`}>{label}</div>
                      {confidence !== null ? (
                        <div className="confidence">
                          <div className="confidence-bar">
                            <div className="confidence-fill" style={{ width: `${Math.round(confidence * 100)}%` }} />
                          </div>
                          <div className="confidence-text">{(confidence * 100).toFixed(1)}% confidence</div>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <pre className="raw">{data && data.result_text ? data.result_text : JSON.stringify(data, null, 2)}</pre>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div>No results yet.</div>
        )}
      </div>
    </div>
  )
}
