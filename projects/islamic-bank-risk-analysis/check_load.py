import sys, os
sys.path.insert(0, os.getcwd())

from backend.models import manager

print(f"Current Dir: {os.getcwd()}")
print(f"Saved Dir: {os.path.join(os.getcwd(), 'backend', 'saved_models')}")

ok = manager.load_from_disk()
print(f"Load Result: {ok}")
print(f"Trained: {manager._trained}")
print(f"Meta: {manager.meta}")
