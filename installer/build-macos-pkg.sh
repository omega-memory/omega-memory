#!/bin/bash
# OMEGA Memory -- macOS .pkg build script
# Downloads python-build-standalone for Apple Silicon and for Intel, installs
# a pinned omega-memory[server] into each, and produces one .pkg. The
# postinstall script keeps the Python that matches the Mac it runs on.
#
# Usage: ./build-macos-pkg.sh [VERSION]
#   VERSION: package and omega-memory version string (default: 1.5.4)
#
# Environment:
#   OMEGA_PKG_BUILD_DIR  where to build (default: installer/build/macos)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${OMEGA_PKG_BUILD_DIR:-$SCRIPT_DIR/build/macos}"
PAYLOAD_DIR="$BUILD_DIR/payload"
PKG_ID="com.omega.memory"
PKG_VERSION="${1:-1.5.4}"
OMEGA_VERSION="$PKG_VERSION"
PYTHON_VERSION="3.12"
PYTHON_PATCH_VERSION="3.12.9"
PYTHON_RELEASE="20250212"
ARCHES="arm64 x86_64"

# The oldest macOS each payload runs on, set by what the binaries themselves
# need (their wheel tags claim older): on Apple Silicon, onnxruntime since
# 1.24 and sqlite-vec need macOS 14; on Intel, sqlite-vec since 0.1.7 needs
# macOS 15 (onnxruntime's last Intel build, 1.23.2, needs 13.4). Packages are
# chosen for these floors rather than for the build machine's macOS, and
# check_payload.py fails the build if any binary needs more.
# macos/Distribution.xml enforces the same numbers: change both together.
min_macos() {
    case "$1" in
        arm64)  echo "14.0" ;;
        x86_64) echo "15.0" ;;
    esac
}

pbs_arch() {
    case "$1" in
        arm64)  echo "aarch64" ;;
        x86_64) echo "x86_64" ;;
    esac
}

# pip runs from the Python that matches the build machine and installs the
# other architecture's packages by platform tag, so no Intel runner is needed.
HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    arm64|x86_64) ;;
    *) echo "ERROR: Unsupported build architecture: $HOST_ARCH"; exit 1 ;;
esac

echo "=== OMEGA macOS Installer Build ==="
echo "omega-memory: $OMEGA_VERSION"
echo "Python: $PYTHON_PATCH_VERSION ($PYTHON_RELEASE)"
for arch in $ARCHES; do
    echo "Payload: $arch (download: $(pbs_arch "$arch"), macOS $(min_macos "$arch") or later)"
done
echo "Build directory: $BUILD_DIR"
echo ""

# --- Clean previous build ---
rm -rf "$BUILD_DIR"
mkdir -p "$PAYLOAD_DIR" "$BUILD_DIR/scripts" "$BUILD_DIR/resources" "$BUILD_DIR/dist"

# --- Step 1: Download and extract python-build-standalone ---
echo "Step 1: Downloading python-build-standalone..."
for arch in $ARCHES; do
    url="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/cpython-${PYTHON_PATCH_VERSION}+${PYTHON_RELEASE}-$(pbs_arch "$arch")-apple-darwin-install_only_stripped.tar.gz"
    tarball="$BUILD_DIR/python-$arch.tar.gz"
    curl -fSL --progress-bar -o "$tarball" "$url"
    mkdir -p "$BUILD_DIR/extract-$arch"
    tar -xzf "$tarball" -C "$BUILD_DIR/extract-$arch"
    mv "$BUILD_DIR/extract-$arch/python" "$PAYLOAD_DIR/python-$arch"
    rm -rf "$tarball" "$BUILD_DIR/extract-$arch"
    echo "  $arch: extracted to $PAYLOAD_DIR/python-$arch/"
done
HOST_PYTHON="$PAYLOAD_DIR/python-$HOST_ARCH/bin/python3"

# --- Step 2: Install omega-memory[server] into each payload ---
echo "Step 2: Installing omega-memory[server]==$OMEGA_VERSION..."
for arch in $ARCHES; do
    floor="$(min_macos "$arch")"
    site="$PAYLOAD_DIR/python-$arch/lib/python$PYTHON_VERSION/site-packages"
    "$HOST_PYTHON" -m pip install --quiet --no-cache-dir \
        --target "$site" \
        --platform "macosx_${floor%%.*}_0_$arch" \
        --only-binary=:all: \
        --implementation cp \
        --python-version "$PYTHON_VERSION" \
        --abi "cp${PYTHON_VERSION/./}" \
        "omega-memory[server]==$OMEGA_VERSION"
    # --target puts console scripts here with the build machine's Python in
    # their shebang. Nothing uses them: Claude Desktop runs
    # `python3 -m omega.server.mcp_server`.
    rm -rf "${site:?}/bin"
    echo "  $arch: installed omega-memory==$OMEGA_VERSION"
done

# --- Step 3: Check every binary runs where the pkg says it does ---
echo "Step 3: Checking payload binaries..."
for arch in $ARCHES; do
    "$HOST_PYTHON" "$SCRIPT_DIR/macos/check_payload.py" \
        "$PAYLOAD_DIR/python-$arch" "$arch" "$(min_macos "$arch")"
done

# --- Step 4: Copy support files ---
echo "Step 4: Copying support files..."
cp "$SCRIPT_DIR/configure_claude.py" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/uninstall-omega.sh" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/setup-instructions.sh" "$PAYLOAD_DIR/"
cp "$SCRIPT_DIR/macos/scripts/postinstall" "$BUILD_DIR/scripts/"
cp "$SCRIPT_DIR/macos/resources/"* "$BUILD_DIR/resources/"
cp "$SCRIPT_DIR/macos/Distribution.xml" "$BUILD_DIR/"
"$HOST_PYTHON" - "$BUILD_DIR/Distribution.xml" "$PKG_VERSION" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
version = sys.argv[2]
text = path.read_text()
text = re.sub(
    r'(<pkg-ref id="com\.omega\.memory"\s+version=")[^"]+(")',
    rf'\g<1>{version}\2',
    text,
    count=1,
)
path.write_text(text)
PY

# --- Step 5: Build component package ---
echo "Step 5: Building component package..."
pkgbuild \
    --identifier "$PKG_ID" \
    --version "$PKG_VERSION" \
    --root "$PAYLOAD_DIR" \
    --install-location "Library/OMEGA" \
    --scripts "$BUILD_DIR/scripts" \
    "$BUILD_DIR/omega-memory.pkg"
echo "  Built component package"

# --- Step 6: Build product archive ---
echo "Step 6: Building product archive..."
productbuild \
    --distribution "$BUILD_DIR/Distribution.xml" \
    --resources "$BUILD_DIR/resources" \
    --package-path "$BUILD_DIR" \
    "$BUILD_DIR/dist/OMEGA-Memory.pkg"
echo "  Built OMEGA-Memory.pkg"

# --- Done ---
PKG_SIZE="$(du -h "$BUILD_DIR/dist/OMEGA-Memory.pkg" | cut -f1)"
echo ""
echo "=== Build complete ==="
echo "Output: $BUILD_DIR/dist/OMEGA-Memory.pkg ($PKG_SIZE)"
echo ""
echo "The pkg is unsigned. To sign and notarize it, see installer/README.md."
