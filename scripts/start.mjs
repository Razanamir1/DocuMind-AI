import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { ensurePythonBackend } from './bootstrap-python.mjs';

const rootDir = process.cwd();
const backendDir = path.join(rootDir, 'backend');
const port = process.env.PORT || '3000';
const pythonBin = ensurePythonBackend(rootDir);

console.log(`[DocuMind AI] Starting Python FastAPI server (${pythonBin}) on 0.0.0.0:${port}`);

const proc = spawn(
  pythonBin,
  [
    '-m',
    'uvicorn',
    'app.main:app',
    '--host',
    '0.0.0.0',
    '--port',
    port,
    '--proxy-headers',
    '--forwarded-allow-ips',
    '*',
  ],
  {
    cwd: backendDir,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: 'inherit',
  }
);

const shutdown = (code = 0) => {
  if (!proc.killed) proc.kill('SIGTERM');
  process.exit(code);
};
process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
proc.on('exit', (code) => process.exit(code ?? 0));
