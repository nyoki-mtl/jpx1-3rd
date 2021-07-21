from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.resolve()

cfg_dir = root_dir / 'configs'
data_dir = root_dir / 'data'
work_dir = root_dir / 'work_dirs'
tool_dir = root_dir / 'tools'
submit_dir = root_dir / 'submit_works'

