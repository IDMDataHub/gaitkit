function installPythonBackend(packageName)
%INSTALLPYTHONBACKEND Install BIKEgait into current Python environment.
%   BIKEgait.installPythonBackend() installs package 'BIKEgait'.

if nargin < 1 || isempty(packageName)
    packageName = 'BIKEgait';
end

if ~(ischar(packageName) || isstring(packageName))
    error('packageName must be char or string');
end

cmd = sprintf('"%s" -m pip install %s', char(pyenv().Executable), char(packageName));
status = system(cmd);
if status ~= 0
    error('pip install failed with status %d', status);
end
end
