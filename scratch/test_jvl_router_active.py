from jvl.core.config import load_user_config, load_config
from jvl.core.router import BackendRouter

user_config = load_user_config()
repo_config = load_config()
router = BackendRouter(user_config, repo_config=repo_config)

print("Active backend:", router.active_backend)
print("Active model:", router.active_model)
client = router._get_client()
print("Client model:", client.model)
print("Client think:", getattr(client, "think", "no think attribute"))
