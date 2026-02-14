function result = detect(method, frames, fps, units)
%DETECT Detect gait events using the Python gaitkit backend.
%   result = gaitkit.detect(method, frames, fps, units)
%
% Inputs
%   method : char/string detector name (e.g. 'bayesian_bis')
%   frames : struct array or cell array of frame structs
%   fps    : scalar sampling frequency in Hz (default: 100)
%   units  : struct with fields .position ('mm'|'m') and .angles ('deg'|'rad')

if nargin < 3
    fps = 100;
end
if nargin < 4 || isempty(units)
    units = struct('position', 'mm', 'angles', 'deg');
end

if ~(ischar(method) || isstring(method))
    error('method must be char or string');
end
if ~isscalar(fps) || ~isnumeric(fps) || ~isfinite(fps) || fps <= 0
    error('fps must be a positive scalar');
end

jsonMod = py.importlib.import_module('json');
pyFrames = jsonMod.loads(jsonencode(frames));
pyUnits = jsonMod.loads(jsonencode(units));
pyResult = py.gaitkit.detect_events_structured(char(method), pyFrames, double(fps), pyUnits);

resultJson = char(jsonMod.dumps(pyResult));
result = jsondecode(resultJson);
end
