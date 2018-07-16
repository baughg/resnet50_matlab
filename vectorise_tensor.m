function Ao = vectorise_tensor( Ai )
[Kh,Kw,Fo,Fi] = size(Ai);

Ao = zeros(1,Kh*Kw*Fo*Fi);
step = Kh * Kw;
loc = 1:step;

for fo = 1:Fo
    for fi = 1:Fi
        p = Ai(:,:,fo,fi)';
        Ao(loc) = p(:);
        loc = loc + step;
    end
end
end

