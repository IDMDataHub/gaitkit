function installPythonBackend(packageName)
%INSTALLPYTHONBACKEND Install gaitkit into current Python environment.
%   gaitkit.installPythonBackend() installs package 'gaitkit'.

if nargin < 1 || isempty(packageName)
    packageName = 'gaitkit';
end

if ~(ischar(packageName) || isstring(packageName))
    error('packageName must be char or string');
end

cmd = sprintf('"%s" -m pip install %s', char(pyenv().Executable), char(packageName));
[status, cmdout] = system(cmd);
if status ~= 0
    error('pip install failed with status %d. Output:\n%s', status, cmdout);
end
end
