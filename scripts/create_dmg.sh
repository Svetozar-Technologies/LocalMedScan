#!/bin/bash
# Create a macOS DMG from the PyInstaller build output.
# Usage: ./scripts/create_dmg.sh

set -e

APP_NAME="LocalMedScan"
VERSION="1.0.0"
DMG_NAME="${APP_NAME}-${VERSION}-macOS.dmg"
BUILD_DIR="dist"
APP_PATH="${BUILD_DIR}/${APP_NAME}.app"

echo "=== Building ${APP_NAME} v${VERSION} ==="

# Step 1: Run PyInstaller
echo "[1/3] Running PyInstaller..."
pyinstaller "${APP_NAME}.spec" --noconfirm

if [ ! -d "${APP_PATH}" ]; then
    echo "ERROR: ${APP_PATH} not found. PyInstaller build failed."
    exit 1
fi

echo "[2/3] Creating DMG..."

# Step 2: Create DMG
if command -v create-dmg &> /dev/null; then
    create-dmg \
        --volname "${APP_NAME}" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "${APP_NAME}.app" 175 190 \
        --app-drop-link 425 190 \
        "${BUILD_DIR}/${DMG_NAME}" \
        "${APP_PATH}"
else
    # Fallback: simple DMG creation
    hdiutil create -volname "${APP_NAME}" \
        -srcfolder "${APP_PATH}" \
        -ov -format UDZO \
        "${BUILD_DIR}/${DMG_NAME}"
fi

echo "[3/3] Done!"
echo "DMG created: ${BUILD_DIR}/${DMG_NAME}"
echo "Size: $(du -h "${BUILD_DIR}/${DMG_NAME}" | cut -f1)"
