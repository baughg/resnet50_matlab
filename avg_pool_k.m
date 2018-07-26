function layer1_mp = avg_pool_k( layer1,ks )

[H,W,C] = size(layer1);

Ho = ceil(H / ks);
Wo = ceil(W / ks);

layer1_mp = zeros(Ho, Wo, C);
Hp = Ho * ks;
Wp = Wo * ks;
chan = ones(Hp,Wp)*NaN;

for c = 1:C
    chan(1:H,1:W) = layer1(:,:,c);
    bx = 1:ks;
    by = 1:ks;
    
    for y = 1:ks:H
        for x = 1:ks:W
            A = chan(by,bx);
            layer1_mp(y,x,c) = mean(A(:));
            bx = bx + ks;
        end
        
        bx = 1:ks;
        by = by + ks;
    end    
end
end

