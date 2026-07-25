#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_DIR=${SCRIPT_DIR:h}
OUTPUT_DIR="$PROJECT_DIR/dist/macos"
BUILD_DIR=$(mktemp -d /private/tmp/frame-splitter-build.XXXXXX)
trap 'rm -rf "$BUILD_DIR"' EXIT
APP_NAME="Frame Splitter"
APP_PATH="$BUILD_DIR/$APP_NAME.app"
CONTENTS_PATH="$APP_PATH/Contents"
RESOURCES_PATH="$CONTENTS_PATH/Resources"
WEBAPP_PATH="$RESOURCES_PATH/WebApp"
ICONSET_PATH="$BUILD_DIR/AppIcon.iconset"
DMG_STAGE_PATH="$BUILD_DIR/dmg-stage"
DMG_PATH="$BUILD_DIR/Frame-Splitter-macOS.dmg"
OUTPUT_DMG_PATH="$OUTPUT_DIR/Frame-Splitter-macOS.dmg"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CONTENTS_PATH/MacOS" "$WEBAPP_PATH/vendor" "$ICONSET_PATH" "$DMG_STAGE_PATH"

xcrun clang \
  "$SCRIPT_DIR/FrameSplitterApp.m" \
  -o "$CONTENTS_PATH/MacOS/FrameSplitter" \
  -fobjc-arc \
  -fmodules-cache-path="$BUILD_DIR/ModuleCache" \
  -mmacosx-version-min=13.0 \
  -framework Cocoa \
  -framework WebKit \
  -O

cp "$SCRIPT_DIR/Info.plist" "$CONTENTS_PATH/Info.plist"
cp "$PROJECT_DIR/docs/index.html" "$WEBAPP_PATH/index.html"
cp "$PROJECT_DIR/docs/favicon.svg" "$WEBAPP_PATH/favicon.svg"
cp "$SCRIPT_DIR/vendor/jszip.min.js" "$WEBAPP_PATH/vendor/jszip.min.js"
cp "$SCRIPT_DIR/vendor/UTIF.js" "$WEBAPP_PATH/vendor/UTIF.js"

sed -i '' \
  's#https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js#vendor/jszip.min.js#g' \
  "$WEBAPP_PATH/index.html"
sed -i '' \
  's#https://cdn.jsdelivr.net/npm/utif@3.1.0/UTIF.js#vendor/UTIF.js#g' \
  "$WEBAPP_PATH/index.html"

xcrun clang \
  "$SCRIPT_DIR/GenerateIcon.m" \
  -o "$BUILD_DIR/GenerateIcon" \
  -fobjc-arc \
  -fmodules-cache-path="$BUILD_DIR/ModuleCache" \
  -mmacosx-version-min=13.0 \
  -framework Cocoa
"$BUILD_DIR/GenerateIcon" "$BUILD_DIR/AppIcon-1024.png"

for spec in \
  "16 icon_16x16.png" \
  "32 icon_16x16@2x.png" \
  "32 icon_32x32.png" \
  "64 icon_32x32@2x.png" \
  "128 icon_128x128.png" \
  "256 icon_128x128@2x.png" \
  "256 icon_256x256.png" \
  "512 icon_256x256@2x.png" \
  "512 icon_512x512.png" \
  "1024 icon_512x512@2x.png"
do
  pixels=${spec%% *}
  filename=${spec#* }
  sips -z "$pixels" "$pixels" "$BUILD_DIR/AppIcon-1024.png" \
    --out "$ICONSET_PATH/$filename" >/dev/null
done

node "$SCRIPT_DIR/CreateIcns.js" "$ICONSET_PATH" "$RESOURCES_PATH/AppIcon.icns"
# Cloud-synced workspaces can attach Finder/provenance metadata while the app
# bundle is being assembled. Ad-hoc signing rejects those extended attributes.
xattr -cr "$APP_PATH"
codesign --force --deep --sign - "$APP_PATH"

cp -R "$APP_PATH" "$DMG_STAGE_PATH/$APP_NAME.app"
ln -s /Applications "$DMG_STAGE_PATH/Applications"
xattr -cr "$DMG_STAGE_PATH/$APP_NAME.app"
codesign --verify --deep --strict "$DMG_STAGE_PATH/$APP_NAME.app"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$DMG_STAGE_PATH" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

cp "$DMG_PATH" "$OUTPUT_DMG_PATH"
echo "$OUTPUT_DMG_PATH"
