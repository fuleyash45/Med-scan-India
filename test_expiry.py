test_expiry.py
from modules.expiry import get_expiry, get_alert

print('expiry.py loaded OK')

tests = ['12/2025', 'JAN-2026', 'MAR-2024', None]
for t in tests:
    status, msg = get_alert(t)
    print(f'{t} -> [{status}] {msg}')