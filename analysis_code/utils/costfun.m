function sse = costfun(params,x_data,y_data)

y_pred = func_piecewise(params,x_data);

regularization = 10000;

sse = sum((y_data-y_pred).^2) / sum((y_data-mean(y_data)).^2);

slope1 = params(1);
ic1 = params(2);
slope2 = params(3);
ic2 = params(4);
breakpoint = params(5);

% if slope1*breakpoint + ic1 ~= slope2*breakpoint + ic2
%     sse = sse + regularization;
% end

if slope1 > slope2
    sse = sse + regularization;
end

end

