function layer1_mp = max_pool( layer1 )

[H,W,C] = size(layer1);

col_odd = 1:2:W;
col_even = 2:2:W;
row_odd = 1:2:H;
row_even = 2:2:H;

f1 = layer1(row_odd,col_odd,:);
f2 = layer1(row_odd,col_even,:);
f3 = layer1(row_even,col_odd,:);
f4 = layer1(row_even,col_even,:);

layer1_mp = zeros(size(f4));

for c = 1:C
    max_val1 = max(f1(:,:,c),f2(:,:,c));
    max_val2 = max(f3(:,:,c),f4(:,:,c));
    max_val = max(max_val1,max_val2);
    layer1_mp(:,:,c) = max_val;
end
end

