function methods = listMethods()
%LISTMETHODS Return available detector methods.

jsonMod = py.importlib.import_module('json');
pyMethods = py.BIKEgait.list_methods();
methodsJson = char(jsonMod.dumps(pyMethods));
methods = string(jsondecode(methodsJson));
end
