import { defineConfig } from '@playwright/test'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '../../..')
const servicePort = process.env.LAYA_E2E_SERVICE_PORT || '11002'
const demoPort = process.env.LAYA_E2E_DEMO_PORT || '11003'

export default defineConfig({
  testDir: './tests',
  workers: 1,
  timeout: 90000,
  expect: { timeout: 15000 },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: { baseURL: `http://127.0.0.1:${demoPort}`, trace: 'retain-on-failure' },
  webServer: [
    {
      command: 'poetry run python -m laya.commands.app serve', cwd: root,
      url: `http://127.0.0.1:${servicePort}/health/ready`, timeout: 180000,
      env: {
        LAYA_MODEL_DIR: process.env.LAYA_E2E_MODEL_DIR || '/workspace/model-bin/MigoXV/laya-multilingual',
        LAYA_DEVICE: process.env.LAYA_E2E_DEVICE || 'cuda:0',
        LAYA_DTYPE: process.env.LAYA_E2E_DTYPE || 'fp16',
        LAYA_HOST: '127.0.0.1', LAYA_PORT: servicePort,
        HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1',
      },
      gracefulShutdown: { signal: 'SIGTERM', timeout: 10000 },
    },
    {
      command: `poetry run uvicorn examples.demo01.app:create_app --factory --host 127.0.0.1 --port ${demoPort}`,
      cwd: root, url: `http://127.0.0.1:${demoPort}`, timeout: 30000,
      env: { LAYA_DEMO_SERVICE_URL: `http://127.0.0.1:${servicePort}` },
      gracefulShutdown: { signal: 'SIGTERM', timeout: 10000 },
    },
  ],
})
