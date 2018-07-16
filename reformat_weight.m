function [ Wo ] = reformat_weight( Wi )
[Kh,Kw,Fo,Fi] = size(Wi);
Wo = zeros(Kh,Kw,Fi,Fo);

for fi = 1:Fi
    for fo = 1:Fo
        Wo(:,:,fi,fo) = Wi(:,:,fo,fi);
    end
end

% Wo = squeeze(Wo);
end

