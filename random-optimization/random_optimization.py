import re
import json
import cPickle
import random

class ConfigFile:
    """
    Routines to generate config.json file
    """

    #: Buffer hold the final JSON
    buffer = None

    #: Parameter counter prevents name conflicts
    paramcounter = {}
    
    #: List of tokens, the number of appearance is important
    tokens = {}

    #: List of parameters fully parsed
    parameters = {}

    def __init__(self):
        self.buffer = '{"language": "PYTHON", "main-file": "cafferun.py", "experiment-name": "cafferun", "likelihood": "GAUSSIAN", "variables" : {'

    def parse_in(self, prototxt):
        """ Parse the contents of prototxt file and fill Spearmint config with optimization variables """
		
        # find optimization tokens in the buffer
        pattern = re.compile('.*OPTIMIZE.*')
        matches = re.findall(pattern, prototxt)

        # parse each token and add it to the Spearmint config file object
        for match in matches:

            # extract name and the parameter description
            (name, param) = match.split('OPTIMIZE')
            name = name.replace(':', '').strip()

            # generate new name to avoid conflicts
            newname = self.newname(name)

            # store the token
            self.tokens[len(self.tokens) + 1] = {'name': newname, 'description': param}

            # unserialize parameter description
            param = json.loads(param)

            # store the parsed parameter
            self.parameters[newname] = param

            # fill the json file buffer with variable descriptions
            if param['type'] == 'INT':
                self.smint(newname, param['min'], param['max'])
            if param['type'] == 'ENUM':
                self.smenum(newname, param['options'])

    def newname(self, name):
        """ To avoid parameter name conflicts we might need to append a counter to a name """
        if self.paramcounter.get(name, None) is None:
            self.paramcounter[name] = 1
        else:
            self.paramcounter[name] += 1
        name = name + '_' + str(self.paramcounter[name])
        return name

    def smint(self, name, min, max):
        """ Generate JSON piece corresponding to Spearmint INT type definition """
        self.buffer += '"%s": { "type": "INT", "size": 1, "min": %d, "max": %d},' % (name, min, max)

    def smenum(self, name, options):
        """ Generate JSON piece corresponding to Spearmint ENUM type definition """
        self.buffer += '"%s": { "type": "ENUM", "size": 1, "options" : [%s] },' % (name, ', '.join([str(x) for x in options]))

    def select_selection(self):
        suggestions = {}
        for p in self.parameters.keys():
            param = self.parameters[p]
            ptype = param['type']
            if 'ENUM' in ptype:
                suggestions[p] = random.choice(param['options'])
            elif 'INT' in ptype:
                suggestions[p] = random.randint(int(param['min']), int(param['max']))
            else:
                print "Unknown parameter in random_optimization.py"
                exit()
        return suggestions

