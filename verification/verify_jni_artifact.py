import argparse
import hashlib
import json
import pathlib
import struct
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('jar', type=pathlib.Path)
parser.add_argument('host')
parser.add_argument('receipt', type=pathlib.Path)
args = parser.parse_args()
names = {'macos-aarch64': 'libcandle_jni.dylib', 'linux-x86_64': 'libcandle_jni.so', 'linux-aarch64': 'libcandle_jni.so', 'windows-x86_64': 'candle_jni.dll'}
entry = f'io/github/kotlinmania/candle/native/{args.host}/{names[args.host]}'
with zipfile.ZipFile(args.jar) as jar:
    data = jar.read(entry)
assert b'Java_io_github_kotlinmania_candle_internal_CandleJni_abiVersion' in data
result = {'jar': str(args.jar), 'entry': entry, 'payload_sha256': hashlib.sha256(data).hexdigest()}
if args.host == 'macos-aarch64':
    assert data[:4] == b'\xcf\xfa\xed\xfe', 'Expected a 64-bit Mach-O'
    assert struct.unpack_from('<I', data, 4)[0] == 0x100000C, 'Expected ARM64'
    count = struct.unpack_from('<I', data, 16)[0]
    offset = 32
    minimum = None
    install_name = None
    for _ in range(count):
        command, size = struct.unpack_from('<II', data, offset)
        if command == 0x32:
            minimum = struct.unpack_from('<I', data, offset + 12)[0]
        if command == 0xD:
            start = struct.unpack_from('<I', data, offset + 8)[0]
            install_name = data[offset + start:offset + size].split(b'\0')[0].decode()
        offset += size
    assert minimum is not None and 0 < minimum <= 15 << 16, f'JNI minimum macOS exceeds 15.0: {minimum}'
    assert install_name == '@rpath/libcandle_jni.dylib', f'Non-relocatable install name: {install_name}'
    result.update(macos_minimum=f'{minimum >> 16}.{minimum >> 8 & 255}.{minimum & 255}', install_name=install_name)
elif args.host.startswith('linux'):
    assert data[:6] == b'\x7fELF\x02\x01'
    assert struct.unpack_from('<H', data, 18)[0] == (183 if args.host.endswith('aarch64') else 62)
else:
    assert data[:2] == b'MZ'
    offset = struct.unpack_from('<I', data, 0x3C)[0]
    assert data[offset:offset + 4] == b'PE\0\0'
    assert struct.unpack_from('<H', data, offset + 4)[0] == 0x8664
    assert struct.unpack_from('<H', data, offset + 24)[0] == 0x20B
    assert struct.unpack_from('<H', data, offset + 22)[0] & 0x2000, 'Expected a DLL'
args.receipt.parent.mkdir(parents=True, exist_ok=True)
args.receipt.write_text(json.dumps(result, indent=2) + '\n')
print(f'Actual packaged JNI payload verified for {args.host}: {result}')
