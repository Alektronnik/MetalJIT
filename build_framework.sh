#!/bin/bash
# ==========================================================================
# build_framework.sh — Genera MetalJIT.xcframework (Apple-grade)
# ==========================================================================
# Requisitos: Xcode 15+, proyecto Xcode en Xcode/
#
# Uso:
#   ./build_framework.sh              # Debug
#   ./build_framework.sh release      # Release
#   ./build_framework.sh release sign # Release + firma + notarizacion
#
# Entorno:
#   DEVELOPER_IDENTITY  Identidad de firma (requerido si sign).
#                       Usa 'security find-identity -v -p codesigning' para listar.
#   NOTARY_PROFILE      Perfil de notarytool (requerido si sign).
#                       Configurar con: xcrun notarytool store-credentials <profile>
#
# Salida: Output/MetalJITCore.xcframework/
#         Output/MetalJIT.xcframework/
#         Output/MetalJIT_Notarized.zip  (si sign)
# ==========================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
XCODE_PROJ="$PROJECT_DIR/Xcode/MetalJIT.xcodeproj"
OUTPUT_DIR="$PROJECT_DIR/Output"
CONFIG="${1:-debug}"
SIGN="${2:-}"

if [ ! -d "$XCODE_PROJ" ]; then
    echo "ERROR: No se encontro Xcode/MetalJIT.xcodeproj"
    echo "Regenera con: cd Xcode && xcodegen generate --spec project.yml"
    exit 1
fi

SCHEME_CORE="MetalJITCore"
SCHEME_SWIFT="MetalJIT"

echo "=============================================="
echo " MetalJIT — Build XCFramework"
echo " Config:  $CONFIG"
echo " Firma:   ${SIGN:-ninguna}"
echo "=============================================="

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

XC_CONFIG="Release"
BUILD_FLAGS="SKIP_INSTALL=NO BUILD_LIBRARY_FOR_DISTRIBUTION=YES"
[ "$CONFIG" = "debug" ] && XC_CONFIG="Debug"

# ==========================================================================
# 1. MetalJITCore.framework
# ==========================================================================
echo "[1/5] Archivando MetalJITCore (macOS)..."
xcodebuild archive \
    -project "$XCODE_PROJ" \
    -scheme "$SCHEME_CORE" \
    -destination "generic/platform=macOS" \
    -archivePath "$OUTPUT_DIR/MetalJITCore.xcarchive" \
    -configuration "$XC_CONFIG" \
    $BUILD_FLAGS 2>&1 | tail -5

# ==========================================================================
# 2. MetalJIT.framework
# ==========================================================================
echo "[2/5] Archivando MetalJIT (macOS)..."
xcodebuild archive \
    -project "$XCODE_PROJ" \
    -scheme "$SCHEME_SWIFT" \
    -destination "generic/platform=macOS" \
    -archivePath "$OUTPUT_DIR/MetalJIT.xcarchive" \
    -configuration "$XC_CONFIG" \
    $BUILD_FLAGS 2>&1 | tail -5

# ==========================================================================
# 3. Crear XCFrameworks
# ==========================================================================
echo "[3/5] Creando XCFrameworks..."
xcodebuild -create-xcframework \
    -archive "$OUTPUT_DIR/MetalJITCore.xcarchive" \
    -framework "MetalJITCore.framework" \
    -output "$OUTPUT_DIR/MetalJITCore.xcframework" 2>&1 | tail -3

xcodebuild -create-xcframework \
    -archive "$OUTPUT_DIR/MetalJIT.xcarchive" \
    -framework "MetalJIT.framework" \
    -output "$OUTPUT_DIR/MetalJIT.xcframework" 2>&1 | tail -3

# ==========================================================================
# 4. Firmar (opcional) — se firma el binario dentro del .xcframework
# ==========================================================================
if [ "$SIGN" = "sign" ]; then
    IDENTITY="${DEVELOPER_IDENTITY:-}"
    if [ -z "$IDENTITY" ]; then
        IDENTITY=$(security find-identity -v -p codesigning 2>/dev/null | \
                   grep 'Developer ID Application' | head -1 | sed 's/.*"\(.*\)".*/\1/' || true)
        if [ -z "$IDENTITY" ]; then
            echo "ERROR: No se encontro identidad Developer ID."
            echo "  Exporta DEVELOPER_IDENTITY o instala un certificado."
            echo "  Lista disponibles: security find-identity -v -p codesigning"
            exit 1
        fi
    fi
    echo "[4/5] Firmando xcframeworks con: $IDENTITY"
    for FW in MetalJITCore MetalJIT; do
        ENTITLEMENTS="$PROJECT_DIR/Xcode/${FW}.entitlements"
        BIN="$OUTPUT_DIR/${FW}.xcframework/macos-arm64/${FW}.framework/Versions/A/${FW}"
        if [ -f "$BIN" ]; then
            codesign --deep --force --verify --verbose \
                --sign "$IDENTITY" \
                --options runtime \
                --timestamp \
                --entitlements "$ENTITLEMENTS" \
                "$BIN"
        fi
        BIN="$OUTPUT_DIR/${FW}.xcframework/macos-arm64/${FW}.framework/${FW}"
        if [ -f "$BIN" ]; then
            codesign --deep --force --verify --verbose \
                --sign "$IDENTITY" \
                --options runtime \
                --timestamp \
                --entitlements "$ENTITLEMENTS" \
                "$BIN"
        fi
    done
else
    echo "[4/5] Firma omitida (usa 'sign' como 2o argumento)"
fi

# ==========================================================================
# 5. Empaquetar y notarizar (si firmo)
# ==========================================================================
if [ "$SIGN" = "sign" ]; then
    echo "[5/5] Empaquetando (DMG) y notarizando..."
    STAGE_DIR="$OUTPUT_DIR/MetalJIT"
    rm -rf "$STAGE_DIR"
    mkdir -p "$STAGE_DIR"
    cp -R "$OUTPUT_DIR/MetalJITCore.xcframework" "$STAGE_DIR/"
    cp -R "$OUTPUT_DIR/MetalJIT.xcframework"     "$STAGE_DIR/"

    DMG_PATH="$OUTPUT_DIR/MetalJIT.dmg"
    rm -f "$DMG_PATH"
    hdiutil create -volname MetalJIT -srcfolder "$STAGE_DIR" -ov -format UDZO "$DMG_PATH" 2>&1 | tail -1
    rm -rf "$STAGE_DIR"

    PROFILE="${NOTARY_PROFILE:-MetalJIT}"
    echo "  Enviando a notarizacion (perfil: $PROFILE)..."
    NOTARY_OUT=$(xcrun notarytool submit "$DMG_PATH" \
        --keychain-profile "$PROFILE" \
        --wait 2>&1)
    echo "$NOTARY_OUT" | tail -8

    if echo "$NOTARY_OUT" | grep -q "status: Accepted"; then
        echo "  Notarizacion aceptada. Aplicando stapler..."
        xcrun stapler staple "$DMG_PATH"
        echo "  DMG grapado listo para distribucion."
    else
        echo "  ATENCION: Notarizacion rechazada. Revisa el log arriba."
        echo "  Para detalles: xcrun notarytool log <id> --keychain-profile \"$PROFILE\""
        exit 1
    fi
else
    echo "[5/5] Notarizacion omitida (usa 'sign' como 2o argumento)"
fi

echo ""
echo "=============================================="
echo " XCFrameworks generados:"
echo "   $OUTPUT_DIR/MetalJITCore.xcframework"
echo "   $OUTPUT_DIR/MetalJIT.xcframework"
if [ "$SIGN" = "sign" ]; then
    echo " Distribuible:"
    echo "   $OUTPUT_DIR/MetalJIT.dmg"
fi
echo "=============================================="
