function Ao = fully_connected( Ai, Cout, fc_name )
[H,W,C] = size(Ai);

fc_w = squeeze(read_array(['model/' fc_name '.0.bin'],C,1,1,Cout))';
fc_b = squeeze(read_array(['model/' fc_name '.1.bin'],Cout,1,1,1));

Ao = sum(repmat(squeeze(Ai)',Cout,1) .* fc_w,2)';
Ao = Ao + fc_b';
end

