function methods = listMethods()
%LISTMETHODS Return available detector methods.

try
    jsonMod = py.importlib.import_module('json');
    gaitkitMod = py.importlib.import_module('gaitkit');
catch ME
    error(['Could not import Python gaitkit. Configure pyenv and install gaitkit ', ...
           'in that interpreter. Original error: %s'], ME.message);
end

try
    pyMethods = gaitkitMod.list_methods();
    methodsJson = char(jsonMod.dumps(pyMethods));
    methods = string(jsondecode(methodsJson));
catch ME
    error('Failed to retrieve methods from Python gaitkit: %s', ME.message);
end
end
