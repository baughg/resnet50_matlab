load('res5c_out.mat','res5c_out');

res5c_pool5 = avg_pool_k( res5c_out,7 );
pool5_fc1000 = fully_connected( res5c_pool5, 1000, 'fc1000' );

pb = softmax(pool5_fc1000);

[pb_s, indx_s] = sort(pb,'descend');

fid = fopen('imagenet1000_clsid_to_human.txt');
imagenet_labels = {};
line_no = 1;
tline = fgetl(fid);
while ischar(tline)
    imagenet_labels{line_no} = tline;
    line_no = line_no + 1;
%     disp(tline)
    tline = fgetl(fid);
end
fclose(fid);

fprintf('***************************************\n');

for r = 1:5
    fprintf('%d. %1.3f %s\n',r ,pb_s(r),imagenet_labels{indx_s(r)});
end