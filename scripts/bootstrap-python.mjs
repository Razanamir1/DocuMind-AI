import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const isWin = process.platform === 'win32';

function runChecked(cmd, args, options = {}) {
  const res = spawnSync(cmd, args, {
    stdio: 'inherit',
    ...options,
  });
  return res.status === 0;
}

function getVenvPythonPath(venvDir) {
  return isWin
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');
}

function canImportFastApi(pythonBin) {
  const probeCode =
    "import importlib.util, sys; mods=('uvicorn','fastapi','chromadb','fitz','google.genai'); sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)";
  const res = spawnSync(pythonBin, ['-c', probeCode], { stdio: 'ignore' });
  return res.status === 0;
}

function hasCompatibleSystemPython(pythonBin) {
  const res = spawnSync(
    pythonBin,
    ['-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'],
    { stdio: 'ignore' }
  );
  return res.status === 0;
}

function canWriteDir(dirPath) {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
    const probe = path.join(dirPath, `.write-probe-${process.pid}`);
    fs.writeFileSync(probe, 'ok', 'utf8');
    fs.unlinkSync(probe);
    return true;
  } catch {
    return false;
  }
}

function isDirWritableForVenv(backendDir, venvDir) {
  if (!canWriteDir(backendDir)) return false;
  try {
    // Use lstatSync so broken symlinks or existing directories are detected accurately
    fs.lstatSync(venvDir);
    return canWriteDir(venvDir);
  } catch {
    return true;
  }
}

function findManagedPython(managedPythonDir) {
  if (!fs.existsSync(managedPythonDir)) return null;
  try {
    const entries = fs.readdirSync(managedPythonDir, { withFileTypes: true });
    const sorted = entries
      .filter((e) => e.isDirectory() && e.name.startsWith('cpython-3.'))
      .sort((a, b) => b.name.localeCompare(a.name));
    for (const entry of sorted) {
      const candidates = isWin
        ? [path.join(managedPythonDir, entry.name, 'python.exe')]
        : [
            path.join(managedPythonDir, entry.name, 'bin', 'python3.11'),
            path.join(managedPythonDir, entry.name, 'bin', 'python3'),
          ];
      for (const candidate of candidates) {
        if (!fs.existsSync(candidate)) continue;
        const probe = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
        if (probe.status === 0) return candidate;
      }
    }
  } catch {
    return null;
  }
  return null;
}

function isPortableVenv(venvPython, managedPythonDir) {
  if (isWin) return fs.existsSync(venvPython);
  if (!fs.existsSync(venvPython) || !fs.existsSync(managedPythonDir)) return false;
  try {
    const realPy = fs.realpathSync(venvPython);
    const realManagedDir = fs.realpathSync(managedPythonDir);
    return realPy.startsWith(realManagedDir + path.sep);
  } catch {
    return false;
  }
}

function tryRepairRelocatedVenv(venvDir, venvPython, managedPythonDir) {
  if (isWin) return false;
  const pyvenvCfg = path.join(venvDir, 'pyvenv.cfg');
  if (!fs.existsSync(pyvenvCfg)) return false;
  const managedPy = findManagedPython(managedPythonDir);
  if (!managedPy) return false;

  try {
    // Repoint bin/python symlink to the bundled standalone interpreter in backend/.uv-python
    const binDir = path.join(venvDir, 'bin');
    if (!fs.existsSync(binDir)) return false;
    try {
      fs.unlinkSync(venvPython);
    } catch {
      // Ignore if missing
    }
    fs.symlinkSync(managedPy, venvPython);

    // Update home = ... in pyvenv.cfg so Python finds its stdlib in the current directory
    const cfgRaw = fs.readFileSync(pyvenvCfg, 'utf8');
    const newHome = path.dirname(managedPy);
    const updatedCfg = /^home\s*=.*$/m.test(cfgRaw)
      ? cfgRaw.replace(/^home\s*=.*$/m, `home = ${newHome}`)
      : `home = ${newHome}\n${cfgRaw}`;
    fs.writeFileSync(pyvenvCfg, updatedCfg, 'utf8');

    return canImportFastApi(venvPython);
  } catch {
    return false;
  }
}

function findUvBinary(extraDirs = []) {
  const uvName = isWin ? 'uv.exe' : 'uv';
  const candidates = [
    ...extraDirs.map((dir) => path.join(dir, uvName)),
    path.join(os.homedir(), '.local', 'bin', uvName),
    path.join(os.homedir(), '.cargo', 'bin', uvName),
    uvName,
  ];
  for (const candidate of candidates) {
    if (candidate.includes(path.sep) && !fs.existsSync(candidate)) continue;
    const probe = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
    if (probe.status === 0) return candidate;
  }
  return null;
}

export function ensurePythonBackend(rootDir, options = {}) {
  const portable = Boolean(options.portable);
  const backendDir = path.join(rootDir, 'backend');
  const defaultVenvDir = path.join(backendDir, '.venv');
  const defaultVenvPython = getVenvPythonPath(defaultVenvDir);
  const defaultManagedPythonDir = path.join(backendDir, '.uv-python');
  const tmpVenvDir = path.join(os.tmpdir(), 'documind-venv');
  const tmpVenvPython = getVenvPythonPath(tmpVenvDir);
  const systemPython = isWin ? 'python' : 'python3';
  const requirementsFile = path.join(backendDir, 'requirements.txt');

  // 1. Fast path (~30ms): existing backend/.venv works (and is self-contained when --portable is requested)
  if (fs.existsSync(defaultVenvPython) && canImportFastApi(defaultVenvPython)) {
    if (!portable || isPortableVenv(defaultVenvPython, defaultManagedPythonDir)) {
      return defaultVenvPython;
    }
    console.log(
      '[DocuMind AI] Rebuilding backend/.venv with a self-contained standalone Python 3.11 for portable container deployment...'
    );
  }

  // 1b. Fast relocation repair (~5ms): workspace moved paths while backend/.uv-python is bundled alongside
  if (tryRepairRelocatedVenv(defaultVenvDir, defaultVenvPython, defaultManagedPythonDir)) {
    if (!portable || isPortableVenv(defaultVenvPython, defaultManagedPythonDir)) {
      return defaultVenvPython;
    }
  }

  if (!portable) {
    // 1c. Reuse previously bootstrapped /tmp fallback venv if backend/.venv was read-only
    if (fs.existsSync(tmpVenvPython) && canImportFastApi(tmpVenvPython)) {
      return tmpVenvPython;
    }

    // 2. System python already has all required backend packages installed
    if (canImportFastApi(systemPython)) {
      return systemPython;
    }
  }

  // Choose writable target directory (falls back to /tmp if container /workspace is read-only at runtime)
  const useLocalBackendDir = isDirWritableForVenv(backendDir, defaultVenvDir);
  const targetVenvDir = useLocalBackendDir ? defaultVenvDir : tmpVenvDir;
  const targetVenvPython = getVenvPythonPath(targetVenvDir);
  const targetManagedPythonDir = useLocalBackendDir
    ? defaultManagedPythonDir
    : path.join(os.tmpdir(), 'documind-uv-python');

  console.log(
    `[DocuMind AI] Bootstrapping Python FastAPI virtual environment in ${path.relative(rootDir, targetVenvDir) || targetVenvDir}...`
  );

  const tmpUvDir = path.join(os.tmpdir(), 'documind-uv-bin');
  const tmpUvCacheDir = path.join(os.tmpdir(), 'documind-uv-cache');
  let installedTmpUv = false;

  try {
    // 3. Prefer uv when available (or required for --portable standalone Python provisioning)
    let uvBin = findUvBinary([tmpUvDir]);
    const hasSysPy311 = hasCompatibleSystemPython(systemPython);

    if (!uvBin && !isWin) {
      const hasPip =
        hasSysPy311 &&
        spawnSync(systemPython, ['-m', 'pip', '--version'], { stdio: 'ignore' }).status === 0;
      const hasVenv =
        hasSysPy311 &&
        spawnSync(systemPython, ['-m', 'venv', '--help'], { stdio: 'ignore' }).status === 0;

      if (portable || !hasSysPy311 || !hasPip || !hasVenv) {
        console.log('[DocuMind AI] Installing uv package manager to provision Python dependencies...');
        fs.mkdirSync(tmpUvDir, { recursive: true });
        const okInstallUv = runChecked(
          'sh',
          ['-c', 'curl -LsSf https://astral.sh/uv/install.sh | sh'],
          {
            cwd: rootDir,
            env: {
              ...process.env,
              UV_INSTALL_DIR: tmpUvDir,
              UV_NO_MODIFY_PATH: '1',
            },
          }
        );
        if (okInstallUv) {
          installedTmpUv = true;
          uvBin = findUvBinary([tmpUvDir]);
        }
      }
    }

    if (uvBin) {
      const uvEnv = {
        ...process.env,
        UV_CACHE_DIR: tmpUvCacheDir,
        UV_PYTHON_INSTALL_DIR: targetManagedPythonDir,
        UV_VENV_CLEAR: '1',
        UV_LINK_MODE: 'copy',
      };

      // Always pass --clear so stale or broken-symlink venv directories are cleanly replaced.
      // Use only-managed standalone Python when building portable images or when system Python 3.11 is absent.
      const useManagedPython = portable || !hasSysPy311;
      const venvArgs = useManagedPython
        ? ['venv', targetVenvDir, '--clear', '--python', '3.11', '--python-preference', 'only-managed']
        : ['venv', targetVenvDir, '--clear', '--python', systemPython];

      const okVenv = runChecked(uvBin, venvArgs, { cwd: rootDir, env: uvEnv });
      if (!okVenv) {
        throw new Error('Failed to create Python virtual environment with uv.');
      }

      const okInstall = runChecked(
        uvBin,
        ['pip', 'install', '--no-cache', '--python', targetVenvPython, '-r', requirementsFile],
        { cwd: rootDir, env: uvEnv }
      );
      if (!okInstall) {
        throw new Error('Failed to install backend/requirements.txt with uv.');
      }

      if (!canImportFastApi(targetVenvPython)) {
        throw new Error('Provisioned Python virtual environment failed FastAPI import verification.');
      }
      return targetVenvPython;
    }

    // 4. Standard python -m venv + pip fallback (typical on Windows / macOS)
    const venvArgs = ['-m', 'venv', '--clear', targetVenvDir];
    const okVenv = runChecked(systemPython, venvArgs, { cwd: rootDir });
    if (!okVenv) {
      throw new Error('Failed to create Python virtual environment with python -m venv.');
    }
    const okPip = runChecked(
      targetVenvPython,
      ['-m', 'pip', 'install', '--no-cache-dir', '-r', requirementsFile],
      { cwd: rootDir }
    );
    if (!okPip) {
      throw new Error('Failed to install backend/requirements.txt with pip.');
    }
    return targetVenvPython;
  } finally {
    // Keep container layers and tmpfs lean by removing ephemeral uv cache / installer artifacts
    try {
      fs.rmSync(tmpUvCacheDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
    if (installedTmpUv) {
      try {
        fs.rmSync(tmpUvDir, { recursive: true, force: true });
      } catch {
        // Ignore cleanup errors
      }
    }
  }
}

const isDirectRun =
  Boolean(process.argv[1]) &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isDirectRun) {
  const portable = process.argv.includes('--portable');
  const resolved = ensurePythonBackend(process.cwd(), { portable });
  console.log(`[DocuMind AI] Python FastAPI environment ready: ${resolved}`);
}
