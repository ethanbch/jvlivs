from jvl.core.config import load_user_config, USER_CONFIG_PATH
import os

print("USER_CONFIG_PATH:", USER_CONFIG_PATH)
print("Exists:", USER_CONFIG_PATH.exists())
if USER_CONFIG_PATH.exists():
    with open(USER_CONFIG_PATH) as f:
        print("Raw file content:\n", f.read())

config = load_user_config()
print("Parsed config:", config)
