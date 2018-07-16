% input data
input_filename = 'test/ILSVRC2017_test_00001917.jpg';
input_image = imread(input_filename);

input_image = uint8(input_image);
figure(1);
image(input_image); axis image; axis off; colormap(gray(256));

input_image = double(input_image);
input_image = imresize(input_image, [224 224], 'bilinear');

eps = 1.0000e-05;
% input_image = 255*ones(224,224,3);


mean_image = squeeze(read_array('model/ResNet_mean.binaryproto.bin',224,224,3,1));
input_image = input_image - mean_image;

% normalise image

% input_image = input_image / 255;

% read weights and biases
conv1_w = read_array('model/conv1.0.bin',7,7,3,64);
conv1_b = read_array('model/conv1.1.bin',64,1,1,1);
bn_conv1_mu = read_array('model/bn_conv1.0.bin',64,1,1,1);
bn_conv1_var = read_array('model/bn_conv1.1.bin',64,1,1,1);
bn_conv1_scale = read_array('model/bn_conv1.2.bin',1,1,1,1);
scale_conv1_gamma = read_array('model/scale_conv1.0.bin',64,1,1,1);
scale_conv1_b = read_array('model/scale_conv1.1.bin',64,1,1,1);

bn_conv1_sigma = sqrt(bn_conv1_var + eps);

conv1_w = reformat_weight(conv1_w);

data = input_image;
data_conv1 = convolution_full( data, conv1_w, conv1_b, 1 ); % conv1
data_conv1 = downsample_with_stride( data_conv1, 2 );

bn_conv1 = batch_norm( data_conv1, bn_conv1_mu, bn_conv1_sigma, bn_conv1_scale );
bn_conv1_scale = scale_tensor( bn_conv1, scale_conv1_gamma, scale_conv1_b );
conv1_relu = relu(bn_conv1_scale);
conv1_relu_pool1 = max_pool_k( conv1_relu,3 );
conv1_relu_pool1 = downsample_with_stride( conv1_relu_pool1, 2 );

% after pool1
conv_stride = ones(1,3);
% res2a
pool1_scale2a_branch1 = resnet_branch( conv1_relu_pool1 , '2a','1', 256, 0, 0, 1, conv_stride);
pool1_scale2a_branch2c = resnet_branch( conv1_relu_pool1 , '2a','2', 64, 64, 256, 3, conv_stride);
res2a_out = elementwise_add_relu(pool1_scale2a_branch1,pool1_scale2a_branch2c);
% res2b
res2a_scale2b_branch2c = resnet_branch( res2a_out , '2b','2', 64, 64, 256, 3, conv_stride);
res2b_out = elementwise_add_relu(res2a_out,res2a_scale2b_branch2c);
% res2c
res2c_scale2b_branch2c = resnet_branch( res2b_out , '2c','2', 64, 64, 256, 3, conv_stride);
res2c_out = elementwise_add_relu(res2b_out,res2c_scale2b_branch2c);
% res3a
conv_stride(1) = 2;
res2c_scale3a_branch1 = resnet_branch( res2c_out , '3a','1', 512, 0, 0, 1, conv_stride);
res2c_scale3a_branch2c = resnet_branch( res2c_out , '3a','2', 128, 128, 512, 3, conv_stride);
res3a_out = elementwise_add_relu(res2c_scale3a_branch1,res2c_scale3a_branch2c);
% res3b
conv_stride(1) = 1;
res3a_scale3b_branch2c = resnet_branch( res3a_out , '3b','2', 128, 128, 512, 3, conv_stride);
res3b_out = elementwise_add_relu(res3a_out,res3a_scale3b_branch2c);
% res3c
res3b_scale3c_branch2c = resnet_branch( res3b_out , '3c','2', 128, 128, 512, 3, conv_stride);
res3c_out = elementwise_add_relu(res3b_out,res3b_scale3c_branch2c);
% res3d
res3c_scale3d_branch2c = resnet_branch( res3c_out , '3d','2', 128, 128, 512, 3, conv_stride);
res3d_out = elementwise_add_relu(res3c_out,res3c_scale3d_branch2c);
% res4a
conv_stride(1) = 2;
res3d_scale4a_branch1 = resnet_branch( res3d_out , '4a','1', 1024, 0, 0, 1, conv_stride);
res3d_scale4a_branch2c = resnet_branch( res3d_out , '4a','2', 256, 256, 1024, 3, conv_stride);
res4a_out = elementwise_add_relu(res3d_scale4a_branch1,res3d_scale4a_branch2c);
% res4b
conv_stride(1) = 1;
res4a_scale4b_branch2c = resnet_branch( res4a_out , '4b','2', 256, 256, 1024, 3, conv_stride);
res4b_out = elementwise_add_relu(res4a_out,res4a_scale4b_branch2c);
% res4c
res4b_scale4c_branch2c = resnet_branch( res4b_out , '4c','2', 256, 256, 1024, 3, conv_stride);
res4c_out = elementwise_add_relu(res4b_out,res4b_scale4c_branch2c);
% res4d
res4c_scale4d_branch2c = resnet_branch( res4c_out , '4d','2', 256, 256, 1024, 3, conv_stride);
res4d_out = elementwise_add_relu(res4c_out,res4c_scale4d_branch2c);
% res4e
res4d_scale4e_branch2c = resnet_branch( res4d_out , '4e','2', 256, 256, 1024, 3, conv_stride);
res4e_out = elementwise_add_relu(res4d_out,res4d_scale4e_branch2c);
% res4f
res4e_scale4f_branch2c = resnet_branch( res4e_out , '4f','2', 256, 256, 1024, 3, conv_stride);
res4f_out = elementwise_add_relu(res4e_out,res4e_scale4f_branch2c);
% res5a
conv_stride(1) = 2;
res4f_scale5a_branch1 = resnet_branch( res4f_out , '5a','1', 2048, 0, 0, 1, conv_stride);
res4f_scale5a_branch2c = resnet_branch( res4f_out , '5a','2', 512, 512, 2048, 3, conv_stride);
res5a_out = elementwise_add_relu(res4f_scale5a_branch1,res4f_scale5a_branch2c);
% res5b
conv_stride(1) = 1;
res5a_scale5b_branch2c = resnet_branch( res5a_out , '5b','2', 512, 512, 2048, 3, conv_stride);
res5b_out = elementwise_add_relu(res5a_out,res5a_scale5b_branch2c);
% res5c
res5b_scale5c_branch2c = resnet_branch( res5b_out , '5c','2', 512, 512, 2048, 3, conv_stride);
res5c_out = elementwise_add_relu(res5b_out,res5b_scale5c_branch2c);


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










