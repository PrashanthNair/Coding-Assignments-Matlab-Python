%
%	GRS.M
%	This Matlab m-file computes the Gibbons-Ross-Shanken F-test of the
%	efficiency of K-benchmark portfolios r1 with respect to N test portfolios r2.
%
function [stat,pval] = GRS(r1,r2)
[T,K] = size(r1);
[T,N] = size(r2);
W1 = inv(cov(r1,1));
SR1 = 1+mean(r1)*W1*mean(r1)';	% 1+SR^2 of tangency portfolio of r1
r = [r1 r2];
W = inv(cov(r,1));
SR = 1+mean(r)*W*mean(r)';		% 1+SR^2 of tangency portfolio of r
stat = ((T-K-N)/N)*(SR/SR1-1);
pval = 1-fcdf(stat,N,T-K-N);
end