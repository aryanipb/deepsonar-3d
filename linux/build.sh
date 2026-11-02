#!/bin/bash
# Package the desktop app: Electron + webapp -> dist/DeepSonar3D-$VERSION.AppImage
set -euo pipefail
cd "$(dirname "$0")"
VERSION=${VERSION:-4.0}; mkdir -p ../dist
A=build/DeepSonar3D.AppDir; rm -rf build; mkdir -p "$A"
cp -r node_modules/electron/dist "$A/app"
mv "$A/app/electron" "$A/app/deepsonar3d"
rm -f "$A/app/resources/default_app.asar"
mkdir -p "$A/app/resources/app"; cp main.js package.json "$A/app/resources/app/"
rsync -a --exclude test --exclude README.md ../webapp/ "$A/app/resources/webapp/"
cp ../web/public/icon.svg "$A/deepsonar3d.svg"
cat > "$A/deepsonar3d.desktop" <<'D'
[Desktop Entry]
Type=Application
Name=DeepSonar 3D
Exec=deepsonar3d
Icon=deepsonar3d
Categories=Science;
D
cat > "$A/AppRun" <<'R'
#!/bin/bash
exec "$(dirname "$(readlink -f "$0")")/app/deepsonar3d" --no-sandbox "$@"
R
chmod +x "$A/AppRun"
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 appimagetool "$A" ../dist/DeepSonar3D-$VERSION.AppImage
sha256sum ../dist/DeepSonar3D-$VERSION.AppImage > ../dist/DeepSonar3D-$VERSION.AppImage.sha256
