from dotenv import load_dotenv
import os

load_dotenv()

DATASETS_PATH = os.getenv("DATASETS_PATH") or os.getenv("DATASETS_ROOT", "")

BAS_RVG1_PATH = os.path.join(DATASETS_PATH, "BAS-RVG1/RVG1_CLARIN")
