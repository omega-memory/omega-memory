#!/bin/bash
# OMEGA Memory -- sign and notarize a CI-built OMEGA-Memory.pkg on a Mac that
# holds the Developer ID certificates.
#
# CI builds the pkg unsigned. Notarization needs more than signing the pkg:
# Apple rejects a pkg unless every executable and library inside it is signed
# with a Developer ID Application certificate, the hardened runtime and a
# secure timestamp. So this script takes the CI pkg apart, signs each binary
# in its payload, rebuilds the pkg from CI's own payload, scripts, package
# info and Distribution, signs it with the Developer ID Installer certificate,
# submits it to Apple, and staples the ticket. Nothing is rebuilt from source,
# so the result carries exactly what CI built.
#
# Usage:
#   installer/macos/sign-and-notarize.sh [--dry-run] [--profile NAME] OMEGA-Memory.pkg
#
#   --dry-run   Take the pkg apart, list what would be signed, rebuild it
#               unsigned to prove the round trip, and print the signing and
#               notarizing commands. Uses no certificate and contacts no one.
#   --profile   notarytool keychain profile (default: omega-notary; create it
#               once with `xcrun notarytool store-credentials`, see
#               installer/README.md).
#
# Environment:
#   OMEGA_APP_IDENTITY        "Developer ID Application: ..." identity that
#                             signs the binaries
#   OMEGA_INSTALLER_IDENTITY  "Developer ID Installer: ..." identity that
#                             signs the pkg
#   Leave them unset to use the one identity of each kind in the keychain.
#
# Output: signed/OMEGA-Memory.pkg next to the input, notarized and stapled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTITLEMENTS="$SCRIPT_DIR/python.entitlements"
APP_IDENTITY="${OMEGA_APP_IDENTITY:-}"
INSTALLER_IDENTITY="${OMEGA_INSTALLER_IDENTITY:-}"
PROFILE="omega-notary"
DRY_RUN=0
INPUT=""

usage() {
    sed -n '14,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --profile) PROFILE="${2:?--profile needs a name}"; shift ;;
        -h|--help) usage ;;
        -*) echo "ERROR: unknown option $1"; usage ;;
        *) INPUT="$1" ;;
    esac
    shift
done
[ -n "$INPUT" ] || usage
[ -f "$INPUT" ] || { echo "ERROR: $INPUT not found"; exit 1; }
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
OUTPUT_DIR="$(dirname "$INPUT")/signed"
OUTPUT="$OUTPUT_DIR/$(basename "$INPUT")"

step() { echo ""; echo "== $*"; }

# Print a command in dry-run mode; run it otherwise.
run() {
    if [ "$DRY_RUN" = 1 ]; then
        printf '  would run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

step "Checking the input and this Mac"
for tool in pkgutil pkgbuild productbuild productsign codesign xcrun file; do
    command -v "$tool" >/dev/null || { echo "ERROR: $tool not found (install Xcode or its command line tools)"; exit 1; }
done
# pkgutil exits non-zero for an unsigned pkg, so read its words, not its status.
signature="$(pkgutil --check-signature "$INPUT" || true)"
if ! grep -q "Status: no signature" <<<"$signature"; then
    echo "ERROR: $INPUT is already signed. Start from the unsigned pkg CI built."
    exit 1
fi
echo "  Input:  $INPUT"
echo "  SHA-256 $(shasum -a 256 "$INPUT" | cut -d' ' -f1)"

# Settle which keychain identity of one kind ("Developer ID Application" or
# "Developer ID Installer") to use: the one asked for, which must exist, or
# else the only one there is. Reads names only; find-identity prints no keys.
resolve_identity() {
    local kind="$1" wanted="$2" names count
    names="$(security find-identity -v | sed -n "s/.*\"\($kind: [^\"]*\)\".*/\1/p" | sort -u)"
    if [ -n "$wanted" ]; then
        if ! grep -qxF "$wanted" <<<"$names"; then
            echo "ERROR: no valid signing identity named \"$wanted\" in the keychain." >&2
            return 1
        fi
        echo "$wanted"
        return 0
    fi
    count="$(grep -c . <<<"$names" || true)"
    if [ "$count" -ne 1 ]; then
        echo "ERROR: expected one \"$kind\" identity in the keychain, found $count." >&2
        echo "Set OMEGA_APP_IDENTITY and OMEGA_INSTALLER_IDENTITY to choose." >&2
        return 1
    fi
    echo "$names"
}
if [ "$DRY_RUN" = 1 ]; then
    APP_IDENTITY="${APP_IDENTITY:-<Developer ID Application identity>}"
    INSTALLER_IDENTITY="${INSTALLER_IDENTITY:-<Developer ID Installer identity>}"
else
    APP_IDENTITY="$(resolve_identity "Developer ID Application" "$APP_IDENTITY")"
    INSTALLER_IDENTITY="$(resolve_identity "Developer ID Installer" "$INSTALLER_IDENTITY")"
    echo "  Binaries signed as: $APP_IDENTITY"
    echo "  Pkg signed as:      $INSTALLER_IDENTITY"
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/omega-sign.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

step "Taking the pkg apart"
pkgutil --expand-full "$INPUT" "$WORK/expanded"
components=("$WORK"/expanded/*.pkg)
if [ "${#components[@]}" -ne 1 ] || [ ! -d "${components[0]}" ]; then
    echo "ERROR: expected exactly one component package inside $INPUT"
    exit 1
fi
COMPONENT="${components[0]}"
COMPONENT_NAME="$(basename "$COMPONENT")"
pkg_info() {
    sed -n "s/.*<pkg-info [^>]* $1=\"\([^\"]*\)\".*/\1/p" "$COMPONENT/PackageInfo" | head -n 1
}
PKG_ID="$(pkg_info identifier)"
PKG_VERSION="$(pkg_info version)"
INSTALL_LOCATION="$(pkg_info install-location)"
if [ -z "$PKG_ID" ] || [ -z "$PKG_VERSION" ] || [ -z "$INSTALL_LOCATION" ]; then
    echo "ERROR: could not read identifier, version and install-location from $COMPONENT_NAME/PackageInfo"
    exit 1
fi
echo "  Component: $COMPONENT_NAME ($PKG_ID $PKG_VERSION, installs to ~/$INSTALL_LOCATION)"

step "Signing the binaries in the payload"
# Found by content, not by name: every Mach-O file must be signed or Apple
# rejects the pkg. `file` adds a "(for architecture ...)" line per slice of a
# universal binary; those are skipped.
libraries=()
executables=()
while IFS=$'\t' read -r path kind; do
    case "$path" in *"(for architecture "*) continue ;; esac
    case "$kind" in
        *Mach-O*executable*) executables+=("$path") ;;
        *Mach-O*) libraries+=("$path") ;;
    esac
done < <(find "$COMPONENT/Payload" -type f -print0 | xargs -0 file -N --separator $'\t')
echo "  ${#libraries[@]} libraries, ${#executables[@]} executables"
if [ "${#libraries[@]}" -eq 0 ] || [ "${#executables[@]}" -eq 0 ]; then
    echo "ERROR: no Python binaries found; is this an OMEGA-Memory.pkg?"
    exit 1
fi
# Libraries first, then the executables that load them. Python executables
# get the entitlement that lets them load libraries pip installs later.
codesign_args=(--force --timestamp --options runtime --sign "$APP_IDENTITY")
if [ "$DRY_RUN" = 1 ]; then
    run codesign "${codesign_args[@]}" "<each of the ${#libraries[@]} libraries>"
    for path in "${executables[@]}"; do
        run codesign "${codesign_args[@]}" --entitlements "$ENTITLEMENTS" "${path#"$COMPONENT/Payload/"}"
    done
else
    for path in "${libraries[@]}"; do
        codesign "${codesign_args[@]}" "$path"
    done
    for path in "${executables[@]}"; do
        codesign "${codesign_args[@]}" --entitlements "$ENTITLEMENTS" "$path"
    done
    for path in "${libraries[@]}" "${executables[@]}"; do
        codesign --verify --strict "$path"
    done
    echo "  Signed and verified $(( ${#libraries[@]} + ${#executables[@]} )) binaries"
fi

step "Rebuilding the pkg from CI's payload"
pkgbuild \
    --identifier "$PKG_ID" \
    --version "$PKG_VERSION" \
    --root "$COMPONENT/Payload" \
    --install-location "$INSTALL_LOCATION" \
    --scripts "$COMPONENT/Scripts" \
    "$WORK/$COMPONENT_NAME" >/dev/null
productbuild \
    --distribution "$WORK/expanded/Distribution" \
    --resources "$WORK/expanded/Resources" \
    --package-path "$WORK" \
    "$WORK/unsigned.pkg" >/dev/null
original_files="$(pkgutil --payload-files "$INPUT" | LC_ALL=C sort | shasum -a 256)"
rebuilt_files="$(pkgutil --payload-files "$WORK/unsigned.pkg" | LC_ALL=C sort | shasum -a 256)"
if [ "$original_files" != "$rebuilt_files" ]; then
    echo "ERROR: the rebuilt pkg does not list the same files as $INPUT"
    exit 1
fi
echo "  Rebuilt; it installs the same $(pkgutil --payload-files "$INPUT" | wc -l | tr -d ' ') paths as the input"

step "Signing the pkg"
run mkdir -p "$OUTPUT_DIR"
run productsign --sign "$INSTALLER_IDENTITY" --timestamp "$WORK/unsigned.pkg" "$OUTPUT"
run pkgutil --check-signature "$OUTPUT"

step "Notarizing (Apple usually answers within minutes)"
if [ "$DRY_RUN" = 1 ]; then
    run xcrun notarytool submit "$OUTPUT" --keychain-profile "$PROFILE" --wait --output-format json
    run xcrun stapler staple "$OUTPUT"
    run xcrun stapler validate "$OUTPUT"
    run spctl --assess --type install -vv "$OUTPUT"
    echo ""
    echo "Dry run complete: nothing was signed or submitted."
    exit 0
fi
if ! xcrun notarytool submit "$OUTPUT" --keychain-profile "$PROFILE" --wait --output-format json \
        > "$WORK/notary.json"; then
    cat "$WORK/notary.json" 2>/dev/null || true
    echo "ERROR: notarytool submit failed. Check the profile exists: xcrun notarytool history --keychain-profile $PROFILE"
    exit 1
fi
submission_id="$(plutil -extract id raw -o - "$WORK/notary.json")"
status="$(plutil -extract status raw -o - "$WORK/notary.json")"
echo "  Submission $submission_id: $status"
if [ "$status" != "Accepted" ]; then
    xcrun notarytool log "$submission_id" --keychain-profile "$PROFILE" "$OUTPUT_DIR/notary-log.json" || true
    echo "ERROR: Apple did not accept the pkg. Its reasons are in $OUTPUT_DIR/notary-log.json"
    exit 1
fi

step "Stapling the ticket and checking Gatekeeper accepts it"
xcrun stapler staple "$OUTPUT"
xcrun stapler validate "$OUTPUT"
spctl --assess --type install -vv "$OUTPUT"

echo ""
echo "Done: $OUTPUT"
echo "SHA-256 $(shasum -a 256 "$OUTPUT" | cut -d' ' -f1)"
