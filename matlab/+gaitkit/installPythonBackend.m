function installPythonBackend(packageName)
%INSTALLPYTHONBACKEND Install gaitkit into current Python environment.
%   gaitkit.installPythonBackend() installs package 'gaitkit'.
%   gaitkit.installPythonBackend("gaitkit[all]") installs optional extras.

if nargin < 1 || isempty(packageName)
    packageName = 'gaitkit';
end

if ~(ischar(packageName) || isstring(packageName))
    error('packageName must be char or string');
end
packageName = strtrim(char(packageName));
if isempty(packageName)
    error('packageName must be non-empty');
end
if isempty(regexp(packageName, '^[A-Za-z0-9._\-\[\],=<>!~]+$', 'once'))
    error(['packageName contains unsupported characters. Use a valid pip ', ...
           'specifier, e.g. "gaitkit" or "gaitkit[all]".']);
end

pe = pyenv();
if isempty(pe.Executable)
    error(['No Python interpreter configured in MATLAB. ', ...
           'Configure one with pyenv("Version","/path/to/python").']);
end

cmd = sprintf('"%s" -m pip install --disable-pip-version-check --no-input %s', ...
              char(pe.Executable), packageName);
[status, cmdout] = system(cmd);
if status ~= 0
    error('pip install failed with status %d. Output:\n%s', status, cmdout);
end
end
