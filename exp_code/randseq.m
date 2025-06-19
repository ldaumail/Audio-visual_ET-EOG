function [seq, mat] = randseq(mat, block)
	n = size(mat, 1);
	if ~exist('block', 'var')
		block = n;
	end
	
	seq = [];
	while length(seq) < n
		seq = [seq, length(seq) + randperm(block)]; %#ok<AGROW>
	end
	seq = seq(1:n);

	idseq(seq) = 1:length(seq);
	mat = [idseq', mat];
end