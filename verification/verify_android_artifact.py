import argparse
import hashlib
import io
import json
import pathlib
import struct
import xml.etree.ElementTree as ET
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('aar', type=pathlib.Path)
parser.add_argument('metadata', type=pathlib.Path)
parser.add_argument('pom', type=pathlib.Path)
parser.add_argument('receipt', type=pathlib.Path)
args = parser.parse_args()
receipt = {'aar': str(args.aar), 'sha256': hashlib.sha256(args.aar.read_bytes()).hexdigest(), 'abis': {}}
with zipfile.ZipFile(args.aar) as aar:
    with zipfile.ZipFile(io.BytesIO(aar.read('classes.jar'))) as classes:
        for name in ('CandleJni', 'CandleJni'):
            assert f'io/github/kotlinmania/candle/internal/{name}.class' in classes.namelist(), name
    for abi, machine in (('arm64-v8a', 183), ('x86_64', 62)):
        data = aar.read(f'jni/{abi}/libcandle_jni.so')
        assert data[:6] == b'\x7fELF\x02\x01', f'{abi}: expected a little-endian 64-bit ELF'
        assert struct.unpack_from('<H', data, 18)[0] == machine, f'{abi}: wrong instruction set'
        offset = struct.unpack_from('<Q', data, 32)[0]
        size, count = struct.unpack_from('<HH', data, 54)
        loads = []
        for index in range(count):
            fields = struct.unpack_from('<IIQQQQQQ', data, offset + index * size)
            if fields[0] == 1:
                assert fields[7] >= 16384 and (fields[3] - fields[2]) % 16384 == 0, f'{abi}: segment is not 16KB-page compatible'
                loads.append(fields[7])
        assert loads, f'{abi}: no loadable segments'
        assert b'Java_io_github_kotlinmania_candle_internal_CandleJni_abiVersion' in data, f'{abi}: missing raw JNI ABI entry point'
        receipt['abis'][abi] = {'machine': machine, 'load_alignments': loads, 'sha256': hashlib.sha256(data).hexdigest()}
metadata = json.loads(args.metadata.read_text())
for usage in ('java-api', 'java-runtime'):
    matches = [v for v in metadata['variants'] if v.get('attributes', {}).get('org.gradle.usage') == usage and v.get('attributes', {}).get('org.gradle.libraryelements') == 'aar']
    assert len(matches) == 1, f'Missing or ambiguous Android {usage} AAR variant'
    variant = matches[0]
    assert variant['attributes']['org.jetbrains.kotlin.platform.type'] == 'androidJvm'
    assert variant['attributes']['org.gradle.jvm.environment'] == 'android'
    assert variant['files'][0]['sha256'] == receipt['sha256'], 'Metadata does not describe this actual AAR'
ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
assert ET.parse(args.pom).getroot().findtext('m:packaging', namespaces=ns) == 'aar'
receipt['component'] = metadata['component']
args.receipt.parent.mkdir(parents=True, exist_ok=True)
args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
print('Actual Android AAR: Kotlin classes, ARM64/x86-64 JNI, 16KB segments, POM and both metadata variants verified')
