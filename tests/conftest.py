import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, ROOT)

os.environ["PAGEINDEX_DATA_DIR"] = tempfile.mkdtemp(prefix="pageindex-tests-")
os.environ["PAGEINDEX_MCP_API_KEY"] = "test-full-key"
os.environ["PAGEINDEX_FIGURE_MODEL"] = "off"
