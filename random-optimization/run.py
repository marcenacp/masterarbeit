import copy, os, string, subprocess
from random_optimization import ConfigFile

def caffe_failed(log_file):
    """Check whether caffe crashed from its log file"""
    lf = open(log_file).read()
    return 'Check failure stack trace' in lf or 'nan loss' in lf

# read in caffe .prototxt files
trainnetfile = 'template_train_val.prototxt'
solverfile = 'template_solver.prototxt'
trainnet = open(trainnetfile, 'r').read()
solver = open(solverfile, 'r').read()

# check that solver has required placeholders
if '"PLACEHOLDER_NET"' not in solver:
    print 'Your solver.prototxt has to have "net: "PLACEHOLDER_NET"" line in it.'
    exit()
if '"PLACEHOLDER_MODEL_STORE"' not in solver:
    print 'Your solver.prototxt has to have "snapshot_prefix: "PLACEHOLDER_MODEL_STORE"" line in it.'
    exit()
    
# inialize templates for output files
tmpl_trainnet = copy.copy(trainnet)
tmpl_solver = copy.copy(solver)

# parse OPTIMIZE tokens in the prototxt files into spearmint config.json
smconfig = ConfigFile()
smconfig.parse_in(trainnet)
smconfig.parse_in(solver)
params = smconfig.parameters.keys()

# generate .prototxt templates
for i in range(1, len(smconfig.tokens) + 1):
    # replace OPTIMIZE{...} with OPTIMIZE_name in the .prototxt template file
    tmpl_trainnet = string.replace(tmpl_trainnet, smconfig.tokens[i]['description'], '_' + smconfig.tokens[i]['name'], 1)
    tmpl_solver = string.replace(tmpl_solver, smconfig.tokens[i]['description'], '_' + smconfig.tokens[i]['name'], 1)

# random optimization loop
max_it = 50
it = len([name for name in os.listdir('logs') if os.path.isfile('logs/'+name)]) # number of log files already produced
while it < max_it:
    # find suggestions for each parameter
    suggestions = smconfig.select_selection()
    
    # replace occurences in new temporary files
    tmp_trainnet = copy.copy(tmpl_trainnet)
    tmp_solver = copy.copy(tmpl_solver)
    log_file = "logs/caffe{}".format(str(it))
    for p in params:
        tmp_trainnet = string.replace(tmp_trainnet, 'OPTIMIZE_' + p, str(suggestions[p]), 1)
        tmp_solver = string.replace(tmp_solver, 'PLACEHOLDER_MODEL_STORE', os.getcwd()+'/snaps/caffe'+str(it), 1)
        tmp_solver = string.replace(tmp_solver, 'PLACEHOLDER_NET', os.getcwd()+'/train_val.prototxt', 1)
        tmp_solver = string.replace(tmp_solver, 'OPTIMIZE_' + p, str(suggestions[p]), 1)
        
    # store temporary files
    with open('train_val.prototxt', 'w') as f:
        f.write(tmp_trainnet)
    with open('solver.prototxt', 'w') as f:
        f.write(tmp_solver)
        
    # launch caffe optimization
    caffe = subprocess.call("$CAFFE_ROOT/build/tools/caffe train -solver solver.prototxt -gpu 2 2>&1 | tee -a "+log_file, shell=True)
    
    # if caffe optimization successful, increment 'it'
    if caffe_failed(log_file):
        os.remove(log_file)
    else:
        it += 1
        

# remove all temporary/template files
os.remove('train_val.prototxt')
os.remove('solver.prototxt')
os.remove('random_optimization.pyc')

