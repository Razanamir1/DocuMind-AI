import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import process from 'node:process';
import { ensurePythonBackend } from './bootstrap-python.mjs';

const rootDir = process.cwd();
const backendDir = path.join(rootDir, 'backend');
const distIndex = path.join(rootDir, 'dist', 'index.html');
const isWin = process.platform === 'win32';

function parseCliArg(flagName) {
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === flagName && args[i + 1]) return args[i + 1];
    if (args[i].startsWith(`${flagName}=`)) return args[i].slice(flagName.length + 1);
  }
  return null;
}

function resolveViteBin() {
  return isWin
    ? path.join(rootDir, 'node_modules', '.bin', 'vite.cmd')
    : path.join(rootDir, 'node_modules', '.bin', 'vite');
}

function isPortAvailable(port, host) {
  return new Promise((resolve) => {
    const tester = net
      .createServer()
      .once('error', (err) => {
        if (err && err.code === 'EAFNOSUPPORT') {
          resolve(true);
        } else {
          resolve(false);
        }
      })
      .once('listening', () => {
        tester.close(() => resolve(true));
      })
      .listen(port, host);
  });
}

async function selectAvailablePort(preferredPort) {
  for (let offset = 0; offset < 20; offset += 1) {
    const candidate = preferredPort + offset;
    const freeV4 = await isPortAvailable(candidate, '127.0.0.1');
    const freeAllV4 = freeV4 && (await isPortAvailable(candidate, '0.0.0.0'));
    const freeAllV6 = freeAllV4 && (await isPortAvailable(candidate, '::'));
    if (freeAllV6) return candidate;
  }
  return preferredPort + 20;
}

async function waitForFastApi(url, timeoutMs = 25000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      // Wait for Uvicorn startup
    }
    await new Promise((r) => setTimeout(r, 350));
  }
  return false;
}

const pythonBin = ensurePythonBackend(rootDir);
const appPort = parseCliArg('--port') || process.env.PORT || '3000';
const appHost = parseCliArg('--host') || '0.0.0.0';
const serveStaticFromFastApi =
  process.env.DISABLE_HMR === 'true' || process.env.SERVE_STATIC_FROM_FASTAPI === 'true';

if (serveStaticFromFastApi) {
  if (!fs.existsSync(distIndex)) {
    console.log('[DocuMind AI] Compiled frontend not found; running vite build...');
    const build = spawnSync(resolveViteBin(), ['build'], {
      cwd: rootDir,
      stdio: 'inherit',
      shell: isWin,
    });
    if (build.status !== 0) {
      process.exit(build.status ?? 1);
    }
  }

  console.log(
    `[DocuMind AI] Starting Python FastAPI server (${pythonBin}) on http://${appHost}:${appPort} (serving compiled React SPA + /api/v1)`
  );

  const backendProc = spawn(
    pythonBin,
    ['-m', 'uvicorn', 'app.main:app', '--host', appHost, '--port', appPort],
    {
      cwd: backendDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: 'inherit',
    }
  );

  const shutdown = (code = 0) => {
    if (!backendProc.killed) backendProc.kill('SIGTERM');
    process.exit(code);
  };
  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));
  backendProc.on('exit', (code) => process.exit(code ?? 0));
} else {
  const preferredFastApiPort = Number(process.env.FASTAPI_PORT || '8000');
  const fastApiPort = String(await selectAvailablePort(preferredFastApiPort));
  if (Number(fastApiPort) !== preferredFastApiPort) {
    console.warn(
      `[DocuMind AI] Port ${preferredFastApiPort} is already in use; starting FastAPI on port ${fastApiPort} instead.`
    );
  }

  console.log(
    `[DocuMind AI] Starting Python FastAPI backend (${pythonBin}) on http://127.0.0.1:${fastApiPort}`
  );

  const backendProc = spawn(
    pythonBin,
    ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', fastApiPort],
    {
      cwd: backendDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: 'inherit',
    }
  );

  let viteProc = null;

  function shutdown(code = 0) {
    if (viteProc && !viteProc.killed) viteProc.kill('SIGTERM');
    if (backendProc && !backendProc.killed) backendProc.kill('SIGTERM');
    process.exit(code);
  }

  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));

  backendProc.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      console.error(`[DocuMind AI] FastAPI exited with code ${code}`);
      shutdown(code);
    }
  });

  const ready = await waitForFastApi(`http://127.0.0.1:${fastApiPort}/api/v1/health`);
  if (!ready) {
    console.warn('[DocuMind AI] FastAPI health check timed out; starting Vite frontend anyway.');
  }

  viteProc = spawn(resolveViteBin(), ['--port', appPort, '--host', appHost], {
    cwd: rootDir,
    env: {
      ...process.env,
      VITE_API_PROXY_TARGET: `http://127.0.0.1:${fastApiPort}`,
    },
    stdio: 'inherit',
    shell: isWin,
  });

  viteProc.on('exit', (code) => {
    shutdown(code ?? 0);
  });
}
