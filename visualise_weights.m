
% fid = fopen('input_image.bin','rb');
% input_image = fread(fid,[28 28],'uint8')';
% fclose(fid);
input_image = imread('test_data/six.png');
input_image = rgb2gray(input_image);
input_image = uint8(input_image);

figure(1);
image(input_image); axis image; axis off; colormap(gray(256));

% scale image
input_image = double(input_image);
input_image = input_image / 255;

% mean_img = read_array('model/mnist_mean.binaryproto.bin',28,28,1,1);
% 
% figure(2);
% imagesc(mean_img); axis image; axis off;


conv1_w = read_array('model/conv1.0.bin',5,5,1,20);
conv1_b = read_array('model/conv1.1.bin',20,1,1,1);

conv2_w = read_array('model/conv2.0.bin',5,5,20,50);
conv2_b = read_array('model/conv2.1.bin',50,1,1,1);

ip1_w = read_array('model/ip1.0.bin',800,1,1,500);
ip1_b = read_array('model/ip1.1.bin',500,1,1,1);

ip2_w = read_array('model/ip2.0.bin',500,1,1,10);
ip2_b = read_array('model/ip2.1.bin',10,1,1,1);

conv1_w_img = weight_image( conv1_w );
conv2_w_img = weight_image( conv2_w );
ip1_w_img = weight_image(ip1_w');
ip2_w_img = weight_image(ip2_w');

imwrite(conv1_w_img,'conv1_w.bmp');
imwrite(conv2_w_img,'conv2_w.bmp');
imwrite(ip1_w_img,'ip1_w.bmp');
imwrite(ip2_w_img,'ip2_w.bmp');
conv1_w = reformat_weight(conv1_w);
conv2_w = reformat_weight(conv2_w);
data = input_image;

data_conv1 = convolution_full( data, conv1_w, conv1_b, 0 );
conv1_pool1 = max_pool( data_conv1 );
pool1_conv2 = convolution_full(conv1_pool1,conv2_w, conv2_b, 0);
conv2_pool2 = max_pool(pool1_conv2);
conv2_pool2_v = vectorise_tensor( conv2_pool2 );

ip1_w_relu1 = sum(repmat(conv2_pool2_v,500,1) .* ip1_w',2)';
ip1_w_relu1 = ip1_w_relu1 + ip1_b';
ip1_w_relu1(ip1_w_relu1 < 0) = 0;

relu1_ip2_w = sum(repmat(ip1_w_relu1,10,1) .* ip2_w',2)';
relu1_ip2_w = relu1_ip2_w + ip2_b';
relu1_ip2_w = relu1_ip2_w ./ max(relu1_ip2_w);
sm_exp = exp(relu1_ip2_w);
sm = sm_exp ./ sum(sm_exp);
digit = find(sm == max(sm)) - 1








