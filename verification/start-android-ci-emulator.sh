#!/usr/bin/env bash
set -euo pipefail
export ANDROID_AVD_HOME="$PWD/build/ci-avd"
mkdir -p "$ANDROID_AVD_HOME"
sdkmanager 'platform-tools' 'emulator' 'system-images;android-35;google_apis;x86_64'
printf 'no\n' | avdmanager create avd --name candle-ci --package 'system-images;android-35;google_apis;x86_64' --force
if [ -e /dev/kvm ]; then sudo chmod 666 /dev/kvm; fi
"$ANDROID_HOME/emulator/emulator" -avd candle-ci -port 5554 -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader_indirect > build/android-ci-emulator.log 2>&1 &
adb -s emulator-5554 wait-for-device
for attempt in $(seq 1 150); do
  if [ "$(adb -s emulator-5554 shell getprop sys.boot_completed | tr -d '\r')" = 1 ]; then exit 0; fi
  sleep 2
done
cat build/android-ci-emulator.log
exit 1
