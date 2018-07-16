function Ao = scale_tensor( Ai, gamma, bias )
[H,W,C] = size(Ai);
Ao = zeros(size(Ai));

for c = 1:C
   Ao(:,:,c) = Ai(:,:,c) * gamma(c);
   Ao(:,:,c) = Ao(:,:,c) + bias(c);
end
end

