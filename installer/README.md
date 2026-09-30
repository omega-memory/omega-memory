# OMEGA Installers

One-click installers for non-technical Claude Desktop users.

- **macOS**: `.pkg` installer (one download for Apple Silicon and Intel)
- **Windows**: `.exe` installer (64-bit)

---

# macOS Installer (.pkg)

## What it does

1. Installs bundled Python 3.12 (python-build-standalone) + a pinned `omega-memory[server]` release to `~/Library/OMEGA`
2. Configures Claude Desktop to use OMEGA as an MCP server
3. No admin privileges required (per-user install)

The pkg carries two Pythons, one built for Apple Silicon and one for Intel,
each with its own copy of OMEGA's packages. The postinstall script keeps the
one that matches the Mac (`sysctl hw.optional.arm64`) and replaces any
previous `~/Library/OMEGA/python` instead of installing over it, so an
upgrade never mixes two versions of the packages.

## Prerequisites

- Apple Silicon Mac with macOS 14 (Sonoma) or later, or
- Intel Mac with macOS 15 (Sequoia) or later

The floors come from the binaries OMEGA depends on, not from their wheel
tags, which claim older: onnxruntime since 1.24 and sqlite-vec need macOS 14
on Apple Silicon, and sqlite-vec since 0.1.7 needs macOS 15 on Intel.
onnxruntime has published no Intel build since 1.23.2, which the Intel
payload therefore uses. `macos/check_payload.py` reads every binary in both
payloads during the build and fails it if one lacks the architecture or needs
a newer macOS; `macos/Distribution.xml` refuses to install below the floors.
The two must change together.
- Claude Desktop installed
- Internet connection (for embedding model download on first use)

## Building locally

### Requirements

- macOS machine
- Internet connection (downloads python-build-standalone for both architectures, ~2 x 60 MB)
- No additional tools needed (uses built-in `pkgbuild`/`productbuild`); an
  Intel runner is not needed, because pip installs the Intel packages by
  platform tag

### Steps

```bash
cd installer
./build-macos-pkg.sh 1.5.20
```

Output: `build/macos/dist/OMEGA-Memory.pkg` (about 175 MB). Set
`OMEGA_PKG_BUILD_DIR` to build somewhere else, such as an external drive; the
build directory needs about 800 MB.

### Automated build

Push a release tag or trigger the `Build macOS Installer` workflow manually in GitHub Actions. The workflow runs on `macos-latest`, builds `OMEGA-Memory.pkg`, checks `omega.__version__` in both payloads (the Intel one under Rosetta when the runner has it), uploads an artifact, and attaches it to `v*` GitHub releases.

The installer is intentionally version-pinned. A `v1.5.4` installer should
install `omega-memory[server]==1.5.4`, not whatever PyPI latest is later.

## Testing checklist

- [ ] Run `OMEGA-Memory.pkg` on a clean macOS install (no Python installed), on Apple Silicon and on Intel
- [ ] Check `lipo -archs ~/Library/OMEGA/python/bin/python3.12` matches the Mac, and `~/Library/OMEGA` has no `python-arm64` or `python-x86_64` left
- [ ] Verify install completes without errors
- [ ] Check `~/Library/OMEGA/python/bin/python3` exists
- [ ] Check `~/Library/Application Support/Claude/claude_desktop_config.json` has `omega-memory` entry
- [ ] Check `.json.bak` backup exists
- [ ] Restart Claude Desktop, verify OMEGA tools appear
- [ ] Say "hello" to Claude, verify `omega_welcome` works
- [ ] Run `~/Library/OMEGA/uninstall-omega.sh`, verify `omega-memory` entry removed
- [ ] Verify `~/.omega` data directory is preserved after uninstall

## Architecture

```
~/Library/OMEGA/                    <- install directory
  python/                           <- python-build-standalone 3.12 for this Mac
    bin/python3                        (the pkg installs python-arm64/ and
                                        python-x86_64/; postinstall keeps one)
    lib/python3.12/site-packages/   <- omega-memory package
  configure_claude.py               <- post-install/uninstall config script
  uninstall-omega.sh                <- uninstall script

~/.omega/                           <- data directory (preserved on uninstall)
  omega.db                          <- memory database
  models/                           <- ONNX embedding model (downloaded on first use)

~/Library/Application Support/Claude/
  claude_desktop_config.json        <- Claude Desktop config (OMEGA entry injected)
  claude_desktop_config.json.bak    <- backup of original config
```

---

# Windows Installer (.exe)

One-click installer (.exe) for non-technical Claude Desktop users on Windows.

## What it does

1. Installs a bundled Python 3.12 + pinned `omega-memory[server]` to `%LOCALAPPDATA%\OMEGA`
2. Configures Claude Desktop to use OMEGA as an MCP server
3. No admin privileges required

## Prerequisites

- Windows 10/11 (64-bit)
- Claude Desktop installed
- Internet connection (for embedding model download on first use)

## Building locally

### Requirements

- Windows machine (or VM)
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) installed
- Internet connection

### Steps

```powershell
# 1. Download Python 3.12 embeddable
mkdir build\python
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip" -OutFile build\python.zip
Expand-Archive build\python.zip -DestinationPath build\python -Force
Remove-Item build\python.zip

# 2. Download get-pip.py
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile build\get-pip.py

# 3. Build installer
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" omega-setup.iss
```

Output: `dist\omega-setup.exe`

### Automated build

Push a release tag or trigger the `Build Windows Installer` workflow manually in GitHub Actions. The workflow installs Inno Setup, downloads embedded Python + `get-pip.py`, builds `omega-setup.exe`, uploads an artifact, and attaches it to `v*` GitHub releases.

The Inno script pins the package version in its `pip install` step. Update
`installer/omega-setup.iss` before each new installer release.

## Testing checklist

- [ ] Run `omega-setup.exe` on a clean Windows VM (no Python installed)
- [ ] Verify install completes without errors
- [ ] Check `%LOCALAPPDATA%\OMEGA\python\python.exe` exists
- [ ] Check `%APPDATA%\Claude\claude_desktop_config.json` has `omega-memory` entry
- [ ] Check `%APPDATA%\Claude\claude_desktop_config.json.bak` backup exists
- [ ] Restart Claude Desktop, verify OMEGA tools appear
- [ ] Say "hello" to Claude, verify `omega_welcome` works
- [ ] Run uninstaller, verify `omega-memory` entry removed from Claude Desktop config
- [ ] Verify `%USERPROFILE%\.omega` data directory is preserved after uninstall

---

# Release checklist

1. Publish and verify `omega-memory` on PyPI.
2. Update installer pins and metadata:
   - `installer/build-macos-pkg.sh` default version
   - `installer/omega-setup.iss` `MyAppVersion`
   - `installer/omega-setup.iss` pinned `pip install omega-memory[server]==...`
3. Build macOS and Windows installers from a `v*` tag or manual workflow.
4. Smoke test both installers on clean machines or VMs.
5. Attach artifacts to the matching GitHub release:
   - `OMEGA-Memory.pkg`
   - `omega-setup.exe`
6. Update website `INSTALLER_VERSION` only after both artifact URLs return 200.

## Architecture

```
%LOCALAPPDATA%\OMEGA\           <- install directory
  python\                       <- Python 3.12 embeddable + site-packages
    python.exe
    Lib\site-packages\omega\    <- omega-memory package
  configure_claude.py           <- post-install/uninstall config script
  get-pip.py                    <- pip bootstrapper (used during install)

%USERPROFILE%\.omega\           <- data directory (preserved on uninstall)
  omega.db                      <- memory database
  models\                       <- ONNX embedding model (downloaded on first use)

%APPDATA%\Claude\
  claude_desktop_config.json    <- Claude Desktop config (OMEGA entry injected)
  claude_desktop_config.json.bak <- backup of original config
```

## Transport

On Windows, the hook server uses TCP `127.0.0.1:19876` instead of Unix domain sockets. The embedding daemon is not used; ONNX models load in-process instead.
