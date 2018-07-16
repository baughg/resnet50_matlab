function layer1_mp = max_pool_k( layer1,ks )

[H,W,C] = size(layer1);




layer1_mp = zeros(size(layer1));
Hp = ceil(H / ks) * ks;
Wp = ceil(W / ks) * ks;
chan = ones(Hp,Wp)*NaN;

for c = 1:C
    chan(1:H,1:W) = layer1(:,:,c);
    bx = 1:ks;
    by = 1:ks;
    
    for y = 1:H
        for x = 1:W
            A = chan(by,bx);
            layer1_mp(y,x,c) = max(A(:));
            bx = bx + 1;
        end
        
        bx = 1:ks;
        by = by + 1;
    end    
end
end

