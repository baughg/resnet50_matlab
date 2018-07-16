function Wght = read_array(filename,W,H,Cn, Cp )
fid = fopen(filename,'rb');

Wght = zeros(W,H,Cn,Cp);

for co = 1:Cp
    weight = zeros(H,W,Cn);
    for ci = 1:Cn
        weight(:,:,ci) = fread(fid,[H W],'float32')';
    end
    
    Wght(:,:,:,co) = weight;
end

% Wght = squeeze(Wght);
fclose(fid);
end

