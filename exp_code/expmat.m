function mat = expmat(varargin)
	mat = [];
	for i = length(varargin):-1:1
		conds = length(varargin{i});
		mat = repmat(mat, conds, 1);
		trials = max([conds, size(mat, 1)]);
		mat = [reshape(repmat(varargin{i}, trials / conds, 1), trials, 1), mat]; %#ok<AGROW>
	end
end