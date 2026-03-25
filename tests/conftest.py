import sys
from pathlib import Path

# 添加项目路径以便导入模块
project_root = Path(__file__).parent.parent / "CV_task" / "赛事题"
sys.path.insert(0, str(project_root))
