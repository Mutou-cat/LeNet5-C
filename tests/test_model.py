"""Run C gradient tests and isolated model/CLI regression checks."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'build/cnn'


def predict(model, pixels):
    return subprocess.run([str(BIN), 'predict', str(model)], input=pixels,
                          capture_output=True, check=False)


with tempfile.TemporaryDirectory() as folder:
    path = Path(folder)
    checkpoint = path / 'roundtrip.cnn'
    subprocess.run([str(ROOT / 'build/test_pool'), str(checkpoint)], check=True)
    raw = bytes(i % 256 for i in range(784))
    good = checkpoint.read_bytes()
    assert good[:8] == b'DLCSPC01'
    assert predict(checkpoint, raw).returncode == 0
    wrong_shape = bytearray(good)
    wrong_shape[8:12] = struct.pack('=i', 28)
    wrong_connections = bytearray(good)
    wrong_connections[8+12*struct.calcsize('=i')+6] ^= 1
    assert len(good) == 8+12*struct.calcsize('=i')+6+96+5726*4
    for label, data, message in [
        ('previous', b'DLCPOOL1'+good[8:], b'Incompatible model'),
        ('connections', wrong_connections, b'connection table'),
        ('old', bytes(15592), b'Incompatible model'),
        ('shape', wrong_shape, b'architecture'),
        ('truncated', good[:-4], b'truncated'),
        ('extra', good+b'x', b'trailing'),
    ]:
        bad = path / (label+'.cnn')
        bad.write_bytes(data)
        result = predict(bad, raw)
        assert result.returncode != 0 and message in result.stderr, result.stderr

    # Real train -> save -> load paths on two synthetic samples (not accuracy evaluation).
    images = path / 'images.idx'
    labels = path / 'labels.idx'
    samples = [raw, bytes(reversed(raw))]
    images.write_bytes(struct.pack('>IIII', 2051, 2, 28, 28)+b''.join(samples))
    labels.write_bytes(struct.pack('>II', 2049, 2)+bytes([0, 1]))
    trained = path / 'trained.cnn'
    subprocess.run([str(BIN), 'train', str(images), str(labels), str(trained)],
                   check=True, capture_output=True)
    wrong = 0
    for label, pixels in enumerate(samples):
        result = predict(trained, pixels)
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert len(output['scores']) == 10
        assert all(-1.71591 <= value <= 1.71591 for value in output['scores'])
        wrong += output['digit'] != label
    result = subprocess.run([str(BIN), 'test', str(images), str(labels), str(trained)],
                            check=True, capture_output=True, text=True)
    assert f'error rate: {wrong * 50:.2f}%' in result.stdout, result.stdout
    print('PASS: version/shape/connections/truncation rejection; train/save/test/predict preprocessing agreement')
