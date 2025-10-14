function F = func_piecewise(params,x)

slope1 = params(1);
ic1 = params(2);
slope2 = params(3);
ic2 = params(4);
breakpoint = params(5);

x1 = x(x<=breakpoint);
x2 = x(x>breakpoint);

y1 = slope1.*x1 + ic1;
% y1 = 0.*x1 + ic1;
y2 = slope2.*x2 + ic2;

F = [y1; y2];

end


