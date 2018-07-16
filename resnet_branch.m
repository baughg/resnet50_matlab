function Ao = resnet_branch( Ai , branch_name_1,branch_name_2,C1,C2,C3, levels, conv_stride)
if levels == 3
    Aa = resnet_branch_op( Ai , branch_name_1,[branch_name_2 'a'],C1, 1, conv_stride(1), 1);
    Ab = resnet_branch_op( Aa , branch_name_1,[branch_name_2 'b'],C2, 3, conv_stride(2), 1);
    Ao = resnet_branch_op( Ab , branch_name_1,[branch_name_2 'c'],C3, 1, conv_stride(3), 0);
elseif levels == 1
    Ao = resnet_branch_op( Ai , branch_name_1,branch_name_2,C1, 1, conv_stride(1), 0);
else
    Ao = Ai;
end
end

