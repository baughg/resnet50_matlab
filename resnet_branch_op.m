function Ao = resnet_branch_op( Ai , branch_name_1,branch_name_2,Cout, kernel_size, conv_stride, do_relu)
[H,W,C] = size(Ai);
eps = 1.0000e-05;

res2a_branch2a_w = read_array(['model/res' branch_name_1 '_branch' branch_name_2 '.0.bin'],kernel_size,kernel_size,C,Cout);
res2a_branch2a_w = reformat_weight(res2a_branch2a_w);

bn2a_branch2a_name = ['model/bn' branch_name_1 '_branch' branch_name_2];
bn2a_branch2a_mu = read_array([bn2a_branch2a_name '.0.bin'],Cout,1,1,1);
bn2a_branch2a_var = read_array([bn2a_branch2a_name '.1.bin'],Cout,1,1,1);
bn2a_branch2a_scale = read_array([bn2a_branch2a_name '.2.bin'],1,1,1,1);
bn2a_branch2a_sigma = sqrt(bn2a_branch2a_var + eps);
scale2a_branch2a_name = ['model/scale' branch_name_1 '_branch' branch_name_2];
scale2a_branch2a_gamma = read_array([scale2a_branch2a_name '.0.bin'],Cout,1,1,1);
scale2a_branch2a_b = read_array([scale2a_branch2a_name '.1.bin'],Cout,1,1,1);
padding = 0;

if kernel_size == 3
    padding = 1;
end

pool1_res2a_branch2a = convolution_full(Ai,res2a_branch2a_w,zeros(Cout,1),padding);

if conv_stride > 1
    pool1_res2a_branch2a = downsample_with_stride( pool1_res2a_branch2a, conv_stride );
end

res2a_branch2a_bn2a_branch2a = batch_norm( pool1_res2a_branch2a, bn2a_branch2a_mu, bn2a_branch2a_sigma, bn2a_branch2a_scale );
bn2a_branch2a_scale2a_branch2a = scale_tensor(res2a_branch2a_bn2a_branch2a,scale2a_branch2a_gamma,scale2a_branch2a_b);

if do_relu > 0
    Ao = relu(bn2a_branch2a_scale2a_branch2a);
else
    Ao = bn2a_branch2a_scale2a_branch2a;
end
end

