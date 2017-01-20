function [ after ] = flatten_backward(layer, after, before)
%FLATTEN_BACKWARD Backward flattening method

data = before.dzdx;
shape = size(data);
output = reshape(data,[64, 7, 7, shape(4)]);
output = permute(output, [3,2,1,4]); % This one is needed to reshape in C mode and not in Fortran mode
after.dzdx = output;

end
