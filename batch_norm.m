function Ao = batch_norm( Ai, mu, sigma, scale_factor )
[H,W,C] = size(Ai);
Ao = zeros(size(Ai));

for c = 1:C
   Ao(:,:,c) = Ai(:,:,c) -  scale_factor*mu(c);
   Ao(:,:,c) = Ao(:,:,c) ./ (scale_factor*sigma(c));
end
end

