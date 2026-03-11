import requests
import json
with open("test_api_out.json", "w") as f:
    eda = requests.get('http://localhost:8000/api/eda-stats').json()
    monte = requests.get('http://localhost:8000/api/monte-carlo/Murabaha').json()
    f.write(json.dumps({
        "eda_keys": list(eda.keys()),
        "has_sector_defaults": "sector_defaults" in eda,
        "monte_keys": list(monte.keys())
    }, indent=2))
