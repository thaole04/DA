from fixedpoint import FixedPoint
scale_qformat = {'m': 2, 'n': 16, 'signed': 0}
val = 0.5
print(f'{FixedPoint(val, **scale_qformat)}')
print(f'{FixedPoint(val, **scale_qformat):04x}')
print(int(f'{FixedPoint(val, **scale_qformat):04x}', 16))
print(int(f'{FixedPoint(val, **scale_qformat):04x}', 16).to_bytes(length=2, byteorder='little'))
