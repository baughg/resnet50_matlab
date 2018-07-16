function Ao = downsample_with_stride( Ai, stride )
[H,W,C] = size(Ai);

h_sel = 1:stride:H;
w_sel = 1:stride:W;

Ho = floor(H / stride);
Wo = floor(W / stride);

Ao = zeros(Ho,Wo,C);

for c = 1:C
   Ao(:,:,c) = Ai(h_sel,w_sel,c);
end
end

