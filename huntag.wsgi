activate_this = '/var/www/huntag_venv/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

import sys
sys.path.append('/var/www/huntag_venv/huntag')

from huntag_rest import add_params, app as application

############################################### MODIFY THIS WHEN NEEDED! ###############################################

model_name = 'test'
cfg_file = 'configs/maxnp.szeged.hfst.yaml'

############################################### MODIFY THIS WHEN NEEDED! ###############################################

add_params(model_name=dev_model_name, cfg_file=dev_cfg_file)

sys.stdout = sys.stderr
